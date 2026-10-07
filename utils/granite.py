"""
utils/granite.py
----------------
IBM Granite LLM client via ibm-watsonx-ai.
Falls back to an offline mock when credentials are absent.
"""

from __future__ import annotations

import os
import textwrap
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("IBM_CLOUD_API_KEY", "")
_URL = os.getenv("IBM_CLOUD_URL", "https://us-south.ml.cloud.ibm.com")
_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "")

# Model ID for IBM Granite 13B Instruct
GRANITE_MODEL_ID = "ibm/granite-13b-instruct-v2"

# Generation parameters
_DEFAULT_PARAMS = {
    "decoding_method": "greedy",
    "max_new_tokens": 1024,
    "min_new_tokens": 10,
    "stop_sequences": [],
    "repetition_penalty": 1.1,
    "temperature": 0.3,
}


def _credentials_present() -> bool:
    return bool(_API_KEY and _PROJECT_ID and _API_KEY != "your_ibm_cloud_api_key_here")


class _MockLLM:
    """Offline mock that returns deterministic placeholder responses."""

    def generate(self, prompt: str) -> str:  # noqa: D401
        snippet = prompt[:120].replace("\n", " ")
        return (
            "[OFFLINE MOCK — No IBM credentials found]\n\n"
            f"Prompt received (first 120 chars): {snippet}...\n\n"
            "To enable real AI responses, copy .env.example → .env and fill in your "
            "IBM_CLOUD_API_KEY, IBM_CLOUD_URL, and IBM_PROJECT_ID."
        )


class GraniteClient:
    """
    Thin wrapper around the ibm-watsonx-ai ModelInference API.
    Automatically degrades to _MockLLM when credentials are missing.
    """

    def __init__(
        self,
        model_id: str = GRANITE_MODEL_ID,
        params: Optional[dict] = None,
    ) -> None:
        self.model_id = model_id
        self.params = params or _DEFAULT_PARAMS
        self._online = False
        self._client = None

        if _credentials_present():
            try:
                from ibm_watsonx_ai import Credentials
                from ibm_watsonx_ai.foundation_models import ModelInference

                creds = Credentials(url=_URL, api_key=_API_KEY)
                self._client = ModelInference(
                    model_id=self.model_id,
                    credentials=creds,
                    project_id=_PROJECT_ID,
                    params=self.params,
                )
                self._online = True
            except Exception as exc:  # pragma: no cover
                print(f"[GraniteClient] Could not initialise watsonx client: {exc}")
                self._client = _MockLLM()
        else:
            self._client = _MockLLM()

    @property
    def is_online(self) -> bool:
        return self._online

    def generate(self, prompt: str) -> str:
        """Send *prompt* to Granite and return the generated text."""
        if self._online:
            result = self._client.generate_text(prompt=prompt)
            return result if isinstance(result, str) else str(result)
        return self._client.generate(prompt)

    def generate_with_system(self, system: str, user: str) -> str:
        """Convenience helper that prepends a system instruction block."""
        full_prompt = (
            f"<|system|>\n{textwrap.dedent(system).strip()}\n"
            f"<|user|>\n{textwrap.dedent(user).strip()}\n"
            "<|assistant|>\n"
        )
        return self.generate(full_prompt)


# Module-level singleton — import and reuse across agents
_default_client: Optional[GraniteClient] = None


def get_client() -> GraniteClient:
    """Return (and lazily initialise) the module-level GraniteClient singleton."""
    global _default_client
    if _default_client is None:
        _default_client = GraniteClient()
    return _default_client
