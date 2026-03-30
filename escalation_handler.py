# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

import aiohttp
from twilio.rest import Client

logger = logging.getLogger(__name__)


async def detect_escalation(va_data: dict) -> Tuple[bool, Optional[dict]]:
    """
    Detects if endSession indicates human escalation.

    Args:
        va_data: The message received from Virtual Agent WebSocket

    Returns:
        (is_escalation, escalation_context) tuple
        - is_escalation: True if this is an escalation to human
        - escalation_context: Dict with escalation details, or None
    """
    if "endSession" not in va_data:
        return False, None

    end_session = va_data["endSession"]

    # Get metadata if present
    metadata = end_session.get("metadata", {})

    # Check for escalation in multiple places
    # 1. Check metadata.session_escalated flag
    # 2. Check top-level reason field
    # 3. Check metadata.reason field
    reason = end_session.get("reason", metadata.get("reason", ""))

    # Check for various escalation indicators
    is_escalation = (
        # Direct escalation flag in metadata
        metadata.get("session_escalated") is True
        # Explicit reason values
        or reason == "escalate_to_human"
        or reason == "human_handoff"
        # Keyword matching in reason text
        or ("escalat" in reason.lower() if reason else False)
        or ("handoff" in reason.lower() if reason else False)
        # Legacy flag
        or end_session.get("requiresHumanAgent", False)
    )

    if not is_escalation:
        return False, None

    # Build escalation context
    escalation_context = {
        "reason": reason or "session_escalated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "context": end_session.get("context", {}),
        "metadata": metadata,
        "params": metadata.get("params", {}),
        "raw_end_session": end_session,  # Include full structure for debugging
    }

    return True, escalation_context


async def send_escalation_webhook(
    webhook_url: str,
    session_id: str,
    escalation_context: dict,
    call_sid: str,
    from_number: str,
    to_number: str,
) -> bool:
    """
    Sends escalation notification to configured webhook.

    Args:
        webhook_url: URL to POST the escalation notification
        session_id: CX session ID
        escalation_context: Escalation details from detect_escalation()
        call_sid: Twilio call SID
        from_number: Caller's phone number
        to_number: Called phone number (the agent's number)

    Returns:
        True if webhook succeeded, False otherwise
    """
    if not webhook_url:
        logger.warning("No webhook URL configured, skipping webhook notification")
        return False

    payload = {
        "event": "escalation_detected",
        "session_id": session_id,
        "call_sid": call_sid,
        "from_number": from_number,
        "to_number": to_number,
        "escalation": escalation_context,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status < 300:
                    logger.info(
                        f"Escalation webhook sent successfully for session {session_id}. "
                        f"Status: {response.status}"
                    )
                    return True
                else:
                    response_text = await response.text()
                    logger.error(
                        f"Escalation webhook failed for session {session_id}. "
                        f"Status: {response.status}, Response: {response_text}"
                    )
                    return False
    except Exception as e:
        logger.error(
            f"Failed to send escalation webhook for session {session_id}: {e}",
            exc_info=True,
        )
        return False


async def transfer_call_to_human(
    base_url: str,
    call_sid: str,
    from_number: str,
    to_number: str,
    escalation_context: dict,
) -> bool:
    """
    Transfers the Twilio call to a human agent using the /transfer endpoint.
    This creates an attended transfer with agent acceptance and context sharing.

    Args:
        base_url: Base URL of the application (e.g., https://your-app.run.app)
        call_sid: The call SID to transfer
        from_number: Caller's phone number
        to_number: Called phone number
        escalation_context: Context to pass to the human agent

    Returns:
        True if transfer succeeded, False otherwise
    """
    if not base_url:
        logger.warning("No BASE_URL configured, skipping call transfer")
        return False

    try:
        # Build context message from escalation data
        metadata = escalation_context.get("metadata", {})
        params = metadata.get("params", {})
        reason = escalation_context.get("reason", "unknown")

        # Build a readable context message
        context_parts = [f"Escalation reason: {reason}"]

        # Add params if present
        if params:
            for key, value in params.items():
                context_parts.append(f"{key}: {value}")

        context_message = ". ".join(context_parts)

        logger.info(
            f"Built context message for transfer: '{context_message}'. "
            f"Reason: '{reason}', Params: {params}"
        )

        # Call the /transfer endpoint
        payload = {
            "CallSid": call_sid,
            "From": from_number,
            "To": to_number,
            "context": context_message,
        }

        transfer_url = f"{base_url}/transfer"

        logger.info(f"Sending transfer request to {transfer_url} with payload: {payload}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                transfer_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status < 300:
                    result = await response.json()
                    logger.info(
                        f"Call {call_sid} transfer initiated successfully. "
                        f"Conference: {result.get('conference_name')}, "
                        f"Agent call: {result.get('agent_call_sid')}"
                    )
                    return True
                else:
                    response_text = await response.text()
                    logger.error(
                        f"Transfer endpoint failed for call {call_sid}. "
                        f"Status: {response.status}, Response: {response_text}"
                    )
                    return False

    except Exception as e:
        logger.error(
            f"Failed to transfer call {call_sid}: {e}",
            exc_info=True,
        )
        return False
