from app.router import ModelRouter


class TestModelRouter:
    def test_select_primary_model(self):
        router = ModelRouter()
        model = router.select_model(complexity="normal")
        assert model == router.settings.primary_model

    def test_select_secondary_for_simple(self):
        router = ModelRouter()
        model = router.select_model(complexity="simple")
        assert model == router.settings.secondary_model

    def test_select_with_valid_preference(self):
        router = ModelRouter()
        primary = router.settings.primary_model
        model = router.select_model(preference=primary)
        assert model == primary

    def test_select_ignores_invalid_preference(self):
        router = ModelRouter()
        model = router.select_model(preference="invalid/model-name")
        assert model == router.settings.primary_model

    def test_available_models(self):
        router = ModelRouter()
        models = router.get_available_models()
        assert len(models) == 2
        assert router.settings.primary_model in models
        assert router.settings.secondary_model in models

    def test_get_logs_empty(self):
        router = ModelRouter()
        logs = router.get_logs()
        assert len(logs) == 0

    def test_prompt_caching_applied_for_claude(self):
        router = ModelRouter()
        messages = [{"role": "system", "content": "You are helpful."}]
        result = router._apply_prompt_caching(messages, "anthropic/claude-3.5-haiku")

        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert result[0]["content"][0]["text"] == "You are helpful."

    def test_prompt_caching_not_applied_for_non_claude(self):
        router = ModelRouter()
        messages = [{"role": "system", "content": "You are helpful."}]
        result = router._apply_prompt_caching(
            messages, "google/gemini-2.0-flash-001"
        )

        assert isinstance(result[0]["content"], str)

    def test_user_messages_not_cached(self):
        router = ModelRouter()
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User question"},
        ]
        result = router._apply_prompt_caching(messages, "anthropic/claude-3.5-haiku")

        # System message gets cache_control
        assert isinstance(result[0]["content"], list)
        # User message stays as plain string
        assert isinstance(result[1]["content"], str)
