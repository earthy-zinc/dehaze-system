from app.service.ai.strategies import agent_config_resolver as resolver


def _stub_db_redis():
    return object(), object()


async def _fake_load_guardrail_defaults(_db, _redis):
    return {
        "prompt_injection": {"enabled": True},
        "pii_mask": {"enabled": True},
        "sensitive_topic": {"enabled": False},
    }


class TestMergeConfig:
    def test_high_priority_overrides_low(self):
        merged = resolver._merge_config(
            {"max_steps": 20, "token_budget": 500000},
            {"max_steps": 30},
            {"token_budget": 1000},
        )
        assert merged == {"max_steps": 30, "token_budget": 1000}

    def test_none_layers_ignored(self):
        merged = resolver._merge_config({"a": 1}, None, {"b": 2})
        assert merged == {"a": 1, "b": 2}

    def test_low_priority_default_kept_when_unoverridden(self):
        merged = resolver._merge_config({"tool_timeout": 60}, {"max_parallel": 3})
        assert merged == {"tool_timeout": 60, "max_parallel": 3}

    def test_empty_override_keeps_defaults(self):
        merged = resolver._merge_config({"max_steps": 20}, {})
        assert merged == {"max_steps": 20}


class TestMergeGuardrails:
    def test_rule_level_override(self):
        defaults = {
            "prompt_injection": {"enabled": True},
            "pii_mask": {"enabled": True},
        }
        merged = resolver._merge_guardrails(defaults, {"prompt_injection": {"enabled": False}})
        assert merged["prompt_injection"]["enabled"] is False
        assert merged["pii_mask"]["enabled"] is True

    def test_session_layer_overrides_agent_layer(self):
        merged = resolver._merge_guardrails(
            {"pii_mask": {"enabled": True}},
            {"pii_mask": {"enabled": True}},
            {"pii_mask": {"enabled": False}},
        )
        assert merged["pii_mask"]["enabled"] is False

    def test_whole_object_and_rule_mixed(self):
        merged = resolver._merge_guardrails(
            {"prompt_injection": {"enabled": True}},
            {"prompt_injection": {"enabled": False}},
        )
        assert merged["prompt_injection"] == {"enabled": False}

    def test_non_dict_layer_skipped(self):
        merged = resolver._merge_guardrails({"pii_mask": {"enabled": True}}, "not-a-dict")
        assert merged["pii_mask"]["enabled"] is True


class TestNestDotted:
    def test_dotted_guardrail_keys_nested(self):
        flat = {"prompt_injection.enabled": True, "pii_mask.enabled": True}
        nested = resolver._nest_dotted(flat)
        assert nested == {
            "prompt_injection": {"enabled": True},
            "pii_mask": {"enabled": True},
        }

    def test_non_dotted_key_kept(self):
        assert resolver._nest_dotted({"max_steps": 20}) == {"max_steps": 20}

    def test_deep_dotted_key(self):
        assert resolver._nest_dotted({"a.b.c": 1}) == {"a": {"b": {"c": 1}}}


class TestResolve:
    async def test_three_level_merge(self, monkeypatch):
        db, redis = _stub_db_redis()
        monkeypatch.setattr(resolver, "load_guardrail_defaults", _fake_load_guardrail_defaults)

        result = await resolver.resolve(
            db,
            redis,
            agent_config={"max_steps": 30, "guardrails": {"pii_mask": {"enabled": False}}},
            conversation_config={"token_budget": 1000},
        )
        assert result["max_steps"] == 30
        assert result["token_budget"] == 1000
        assert result["max_parallel"] == resolver.REASONING_DEFAULTS["max_parallel"]
        assert result["guardrails"]["pii_mask"]["enabled"] is False
        assert result["guardrails"]["prompt_injection"]["enabled"] is True

    async def test_reasoning_defaults_from_constants(self, monkeypatch):
        db, redis = _stub_db_redis()
        monkeypatch.setattr(resolver, "load_guardrail_defaults", _fake_load_guardrail_defaults)

        result = await resolver.resolve(db, redis, agent_config=None, conversation_config=None)
        for key, value in resolver.REASONING_DEFAULTS.items():
            assert result[key] == value

    async def test_guardrails_not_leak_into_reasoning(self, monkeypatch):
        db, redis = _stub_db_redis()
        monkeypatch.setattr(resolver, "load_guardrail_defaults", _fake_load_guardrail_defaults)

        result = await resolver.resolve(
            db,
            redis,
            agent_config={"guardrails": {"pii_mask": {"enabled": False}}},
            conversation_config=None,
        )
        assert "guardrails" not in {k for k in result if k != "guardrails"}
        assert result["token_budget"] == resolver.REASONING_DEFAULTS["token_budget"]
        assert result["guardrails"]["pii_mask"]["enabled"] is False
        assert isinstance(result["guardrails"], dict)

    async def test_guardrail_defaults_cached_in_redis(self, monkeypatch, mock_redis):
        db = _stub_db_redis()[0]

        call_count = [0]

        async def counting_load(_db, type_code):
            call_count[0] += 1
            return {"prompt_injection.enabled": True, "pii_mask.enabled": True}

        monkeypatch.setattr(resolver, "_load_dict_values", counting_load)

        d1 = await resolver.load_guardrail_defaults(db, mock_redis)
        d2 = await resolver.load_guardrail_defaults(db, mock_redis)
        assert d1 == {
            "prompt_injection": {"enabled": True},
            "pii_mask": {"enabled": True},
        }
        assert d1 == d2
        assert call_count[0] == 1
