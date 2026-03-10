from typing import Optional
from app.config import config


class OpenAINotConfigured(Exception):
    """Raised when OpenAI client cannot be initialized due to missing config or missing SDK."""
    pass


class OpenAIClient:
    """
    Server-side OpenAI client wrapper.

    Responsibilities:
        - Validate server configuration (API key, model, timeout).
        - Provide two public behaviors:
            1) explain(): deterministic "explain model/preset/parameter" responses.
            2) ask_question(): free-text Q&A mode for user questions.

    Notes:
        - Network calls are executed via the OpenAI SDK.
        - Errors from the SDK are surfaced as RuntimeError from _chat().
    """

    def __init__(self) -> None:
        """
        Initialize the OpenAI SDK client using server configuration.

        Returns:
            None

        Raises:
            OpenAINotConfigured: If OPENAI_API_KEY is missing or OpenAI SDK is not installed.
        """
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
        Generate a short explanation for a model, a preset, or a single hyperparameter.

        Routing rules:
            - If param_key is None -> explain the model type (MODEL mode).
            - If param_key looks like a preset name -> explain the preset (PRESET mode).
            - Otherwise -> explain the hyperparameter (PARAM mode).

        Args:
            model_type: Model identifier (e.g., "linear", "logistic", "random_forest").
            param_key: None for model explanation, or preset/parameter key.

        Returns:
            str: Assistant explanation text.

        Raises:
            RuntimeError: If the OpenAI request fails or returns an empty response.
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
        """
        Ask a free-text question (optionally scoped to a specific model type).

        Builds a prompt that includes the user's question plus an optional model_type hint,
        then delegates the actual OpenAI call to `_chat()`.

        Args:
            question: Free-text user question to send to the model.
            model_type: Optional model identifier (e.g., "linear", "logistic") to provide context.

        Returns:
            str: The assistant's response text (non-empty, stripped).

        Raises:
            RuntimeError: If the underlying OpenAI request fails or returns an empty response.
        """

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
        """
        Execute a single OpenAI chat completion request.

        Args:
            prompt: User prompt content.
            system: System instruction that sets assistant behavior/style.

        Returns:
            str: Response text from the first choice (non-empty, stripped).

        Raises:
            RuntimeError: If the OpenAI SDK call fails (network/timeout/etc.) or returns empty text.
        """

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
