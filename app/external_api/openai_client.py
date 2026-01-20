from typing import Optional
from app.config import config


class OpenAINotConfigured(Exception):
    pass


class OpenAIClient:
    """
    Server-side OpenAI client.

    Two separated behaviors:
    1) explain_param(): parameter-only explanation (NO context)
    2) ask_question(): free-text question about the model/metrics/results (context/question required)
    """

    def __init__(self) -> None:
        self.api_key = config.OPENAI_API_KEY
        self.model = config.OPENAI_MODEL
        self.timeout = config.OPENAI_TIMEOUT

        if not self.api_key:
            raise OpenAINotConfigured("OPENAI_API_KEY is not configured on the server.")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise OpenAINotConfigured(
                "OpenAI SDK is not installed. Run `pip install openai` on the server."
            ) from e

        self._sdk = OpenAI(api_key=self.api_key)


    def explain(self, model_type: str, param_key: Optional[str]) -> str:
        """
        Unified explanation logic:
        - If param_key == model_type → explain MODEL
        - If param_key looks like a preset → explain PRESET
        - Else → explain PARAMETER
        """

        # --------------------
        # CASE 1: MODEL
        # --------------------
        if param_key is None:
            prompt = (
                f"Explain what the `{model_type}` model is used for.\n"
                "Describe when it is appropriate, its strengths, limitations, "
                "and typical use cases. Keep it concise."
            )

            system = (
                "You are a succinct machine learning tutor. "
                "Explain models clearly and practically."
            )

            return self._chat(prompt, system)

        # --------------------
        # CASE 2: PRESET
        # --------------------
        is_preset = "(" in param_key or ")" in param_key or " " in param_key

        if is_preset:
            prompt = (
                f"Explain the `{param_key}` configuration for the `{model_type}` model.\n"
                "Describe what this preset changes internally and when it should be used.\n"
                "Focus on speed vs accuracy trade-offs."
            )

            system = (
                "You are a succinct ML tutor. "
                "Explain presets clearly and practically."
            )

            return self._chat(prompt, system)

        # --------------------
        # CASE 3: PARAMETER
        # --------------------
        prompt = (
            f"Explain the `{param_key}` parameter for the `{model_type}` model.\n"
            "Cover what it controls, typical values, and trade-offs. Keep it concise."
        )

        system = (
            "You are a succinct ML tutor. "
            "explain the parameter with practical guidance."
        )

        return self._chat(prompt, system)

    def ask_question(self, question: str, model_type: Optional[str] = None) -> str:
        prompt = (
            f"User question:\n{question}\n\n"
            f"Model type (if relevant): {model_type or 'not specified'}\n"
            "Answer clearly and practically."
        )

        return self._chat(
            prompt,
            system=(
                "You are a practical ML assistant. "
                "Provide concise, actionable guidance."
            ),
        )

    def _chat(self, prompt: str, system: str) -> str:
        try:
            resp = self._sdk.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.25,
                timeout=self.timeout,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}") from e

        if not text:
            raise RuntimeError("Empty response from OpenAI.")
        return text