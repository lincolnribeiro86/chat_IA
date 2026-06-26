"""Claude subscription auth via claude-code-sdk.
Requires: pip install claude-code-sdk
Requires CLAUDE_CODE_OAUTH_TOKEN env var (from `claude setup-token`).
"""
import asyncio
import os
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)

_MODEL_MAP = {
    "claude-sub-sonnet": "claude-sonnet-4-6",
    "claude-sub-opus": "claude-opus-4-8",
}


class ClaudeSubProvider:
    """Async iterator wrapper for the claude-code-sdk."""

    _IS_CLAUDE_SUB = True

    def __init__(self, model_id: str, token: str):
        self.claude_model = _MODEL_MAP.get(model_id, "claude-sonnet-4-6")
        self.token = token

    async def astream_text(self, messages: list[dict]) -> AsyncIterator[str]:
        try:
            from claude_code_sdk import query, ClaudeCodeOptions
        except ImportError:
            raise RuntimeError(
                "claude-code-sdk not installed. Run: pip install claude-code-sdk"
            )

        os.environ.setdefault("CLAUDE_CODE_OAUTH_TOKEN", self.token)

        # Build a single prompt from the messages list
        system = ""
        conversation = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system = content
            else:
                conversation.append(f"{'Human' if role == 'user' else 'Assistant'}: {content}")

        prompt = "\n\n".join(conversation)
        if system:
            prompt = f"System: {system}\n\n{prompt}"

        options = ClaudeCodeOptions(model=self.claude_model)

        async for msg in query(prompt=prompt, options=options):
            # The SDK yields AssistantMessage / ResultMessage objects
            msg_type = type(msg).__name__
            if msg_type == "AssistantMessage":
                for block in getattr(msg, "content", []):
                    if hasattr(block, "text"):
                        yield block.text
            elif msg_type == "ResultMessage":
                break


def build_claude_sub(model_id: str, token: str) -> ClaudeSubProvider:
    return ClaudeSubProvider(model_id, token)
