from pathlib import Path

_ROOTS = (
    Path("/data/workspace/dehaze-system/config/sql/data"),
    Path("config/sql/data"),
    Path("/data/workspace/dehaze-system/dehaze-python/../config/sql/data"),
)


def _find_sql(name: str) -> Path:
    for root in _ROOTS:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到种子文件 {name}（尝试过 {_ROOTS}）")


REASONING_KEYS = {
    "max_steps_react": "20",
    "max_steps_plan": "30",
    "max_steps_reflexion": "15",
    "max_iterations_reflexion": "3",
    "reflexion_threshold": "0.8",
    "max_parallel": "5",
    "tool_timeout": "60",
    "token_budget": "500000",
    "retry_max": "2",
}

GUARDRAIL_KEYS = {
    "prompt_injection.enabled": "true",
    "unauthorized_access.enabled": "true",
    "sensitive_topic.enabled": "false",
    "pii_mask.enabled": "true",
    "fact_check.enabled": "false",
    "format_check.enabled": "false",
}

HEALTH_KEYS = {
    "error_rate_warn": "0.1",
    "error_rate_open": "0.3",
    "min_window_calls": "20",
    "consecutive_failures": "5",
    "circuit_cooldown": "60",
}


def _split_outer_commas(s: str) -> list[str]:
    parts, buf, depth = [], [], 0
    for ch in s:
        if ch == "'":
            depth ^= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return parts


def _parse_sys_dict_rows(sql: str) -> list[tuple[str, str, str]]:
    rows = []
    for line in sql.splitlines():
        line = line.strip()
        if not line.startswith("("):
            continue
        line = line.split(";")[0].rstrip(",")
        if not line.startswith("("):
            continue
        inner = line.strip("()").strip()
        parts = _split_outer_commas(inner)
        if len(parts) < 4:
            continue
        type_code = parts[1].strip().strip("'")
        name = parts[2].strip().strip("'")
        value = parts[3].strip().strip("'")
        rows.append((type_code, name, value))
    return rows


def _load_sys_dict_rows() -> list[tuple[str, str, str]]:
    return _parse_sys_dict_rows(_find_sql("sys_dict.sql").read_text(encoding="utf-8"))


class TestSeedCompleteness:
    def test_sys_dict_exists(self):
        assert _find_sql("sys_dict.sql").exists()

    def test_reasoning_defaults_all_present(self):
        rows = _load_sys_dict_rows()
        reasoning = {name: value for tc, name, value in rows if tc == "ai_reasoning_defaults"}
        for key, value in REASONING_KEYS.items():
            assert key in reasoning, f"缺失推理默认 {key}"
            assert reasoning[key] == value, f"{key} 默认值应为 {value}，实际 {reasoning[key]}"
        assert len(reasoning) == 9, f"推理默认应为 9 项，实际 {len(reasoning)}"

    def test_guardrail_defaults_all_present(self):
        rows = _load_sys_dict_rows()
        guardrails = {name: value for tc, name, value in rows if tc == "ai_guardrail_defaults"}
        for key, value in GUARDRAIL_KEYS.items():
            assert key in guardrails, f"缺失护栏默认 {key}"
            assert guardrails[key] == value, f"{key} 默认值应为 {value}"
        assert len(guardrails) == 6, f"护栏默认应为 6 项，实际 {len(guardrails)}"

    def test_provider_health_defaults_all_present(self):
        rows = _load_sys_dict_rows()
        health = {name: value for tc, name, value in rows if tc == "ai_provider_health"}
        for key, value in HEALTH_KEYS.items():
            assert key in health, f"缺失供应商健康阈值 {key}"
            assert health[key] == value, f"{key} 默认值应为 {value}，实际 {health[key]}"
        assert len(health) == 5, f"供应商健康阈值应为 5 项，实际 {len(health)}"

    def test_sys_dict_type_registered(self):
        sql = _find_sql("sys_dict_type.sql").read_text(encoding="utf-8")
        assert "ai_reasoning_defaults" in sql
        assert "ai_guardrail_defaults" in sql
        assert "ai_provider_health" in sql
        assert "ai_embedding" in sql

    def test_guardrail_dotted_keys_are_nested(self):
        from app.service.ai.agent_config_resolver import _nest_dotted

        raw = {"prompt_injection.enabled": "true", "pii_mask.enabled": "false"}
        assert _nest_dotted(raw) == {
            "prompt_injection": {"enabled": "true"},
            "pii_mask": {"enabled": "false"},
        }


class TestEmbeddingSeedContract:
    EMBEDDING_DEFAULTS = {
        "provider_code": "openai",
        "model": "text-embedding-3-small",
        "dims": "1536",
    }

    def test_embedding_defaults_all_present(self):
        rows = _load_sys_dict_rows()
        embedding = {name: value for tc, name, value in rows if tc == "ai_embedding"}
        assert len(embedding) == 3, f"ai_embedding 应为 3 项，实际 {len(embedding)}"
        for key, value in self.EMBEDDING_DEFAULTS.items():
            assert key in embedding, f"缺失 embedding 配置 {key}"
            assert embedding[key] == value, (
                f"{key} 种子值 {embedding[key]} 与代码回落默认 {value} 不一致"
            )

    def test_embedding_dims_matches_es_mapping(self):
        from app.infrastructure.es import ai_memory_index

        assert ai_memory_index.DEFAULT_DIMS == 1536
        assert ai_memory_index.DEFAULT_MODEL == "text-embedding-3-small"
