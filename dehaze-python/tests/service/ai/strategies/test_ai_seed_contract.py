from pathlib import Path


def _find_sql(name: str) -> Path:
    """在仓库根的 config/sql/data 下定位种子 SQL（基于测试文件位置，不依赖 cwd）"""
    for base in Path(__file__).resolve().parents:
        sql_dir = base / "config" / "sql" / "data"
        if (sql_dir / name).exists():
            return sql_dir / name
    raise FileNotFoundError(f"未找到种子文件 {name}（config/sql/data 不在仓库内）")


REASONING_DEFAULTS_EXPECTED = {
    "max_steps_react": 20,
    "max_steps_plan": 30,
    "max_steps_reflexion": 15,
    "max_iterations_reflexion": 3,
    "reflexion_threshold": 0.8,
    "max_parallel": 5,
    "tool_timeout": 60,
    "token_budget": 500000,
    "retry_max": 2,
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

    def test_reasoning_defaults_are_code_constants(self):
        """推理参数默认值不入 sys_dict，唯一来源为 resolver 代码常量。"""
        from app.service.ai.strategies.agent_config_resolver import REASONING_DEFAULTS

        assert REASONING_DEFAULTS == REASONING_DEFAULTS_EXPECTED
        rows = _load_sys_dict_rows()
        assert all(tc != "ai_reasoning_defaults" for tc, _, _ in rows), (
            "推理参数不应存在 sys_dict 种子（默认值为代码常量）"
        )

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
        assert "ai_reasoning_defaults" not in sql
        assert "ai_guardrail_defaults" in sql
        assert "ai_provider_health" in sql
        assert "ai_embedding" in sql
        assert "member_growth_rules" in sql
        assert "favorite_capacity" in sql
        assert "ai_eval" in sql

    def test_guardrail_dotted_keys_are_nested(self):
        from app.service.ai.strategies.agent_config_resolver import _nest_dotted

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


class TestMemberGrowthRulesSeedContract:
    """会员成长值规则（member_growth_rules）种子与消费方默认值一致性。"""

    GROWTH_DEFAULTS = {
        "sign_in_value": "3",
        "sign_in_streak_bonus": "20",
        "rating_growth_value": "5",
        "rating_growth_daily_limit": "5",
    }

    def test_growth_rules_all_present(self):
        rows = _load_sys_dict_rows()
        rules = {name: value for tc, name, value in rows if tc == "member_growth_rules"}
        assert len(rules) == 4, f"member_growth_rules 应为 4 项，实际 {len(rules)}"
        for key, value in self.GROWTH_DEFAULTS.items():
            assert key in rules, f"缺失成长值规则 {key}"
            assert rules[key] == value, f"{key} 种子值 {rules[key]} 与默认 {value} 不一致"

    def test_consumers_fallback_defaults_match_seeds(self):
        """消费方代码回退默认值与 SQL 种子一致（缺键回退不漂移）。"""
        from app.service.feedback_service import (
            RATING_DAILY_GROWTH_LIMIT_DEFAULT,
            RATING_GROWTH_VALUE_DEFAULT,
        )
        from app.service.member.growth_service import (
            SIGN_IN_BASE_GROWTH_DEFAULT,
            SIGN_IN_BONUS_GROWTH_DEFAULT,
        )

        assert str(SIGN_IN_BASE_GROWTH_DEFAULT) == self.GROWTH_DEFAULTS["sign_in_value"]
        assert str(SIGN_IN_BONUS_GROWTH_DEFAULT) == self.GROWTH_DEFAULTS["sign_in_streak_bonus"]
        assert str(RATING_GROWTH_VALUE_DEFAULT) == self.GROWTH_DEFAULTS["rating_growth_value"]
        assert (
            str(RATING_DAILY_GROWTH_LIMIT_DEFAULT)
            == self.GROWTH_DEFAULTS["rating_growth_daily_limit"]
        )


class TestEvalSeedContract:
    """AI 评测质量参数（ai_eval）种子与消费方回退默认值一致性。"""

    EVAL_DEFAULTS = {
        "regression_threshold": "5",
        "judge_consistency_threshold": "90",
        "judge_review_ratio": "1",
    }

    def test_eval_defaults_all_present(self):
        rows = _load_sys_dict_rows()
        evals = {name: value for tc, name, value in rows if tc == "ai_eval"}
        assert len(evals) == 3, f"ai_eval 应为 3 项，实际 {len(evals)}"
        for key, value in self.EVAL_DEFAULTS.items():
            assert key in evals, f"缺失评测参数 {key}"
            assert evals[key] == value, f"{key} 种子值 {evals[key]} 与默认 {value} 不一致"

    def test_consumers_fallback_defaults_match_seeds(self):
        """评测中心消费方回退默认值与 SQL 种子一致（缺键回退不漂移）。"""
        from app.service.ai_eval_center_service import (
            CONSISTENCY_THRESHOLD_DEFAULT,
            REGRESSION_THRESHOLD_DEFAULT,
            REVIEW_RATIO_DEFAULT,
        )

        assert str(REGRESSION_THRESHOLD_DEFAULT) == self.EVAL_DEFAULTS["regression_threshold"]
        assert (
            str(CONSISTENCY_THRESHOLD_DEFAULT)
            == self.EVAL_DEFAULTS["judge_consistency_threshold"]
        )
        assert str(REVIEW_RATIO_DEFAULT) == self.EVAL_DEFAULTS["judge_review_ratio"]


class TestFavoriteCapacitySeedContract:
    """收藏容量（favorite_capacity）种子与消费方默认值/等级映射一致性。"""

    CAPACITY_DEFAULTS = {
        "default": "200",
        "vip1": "500",
        "vip2": "1000",
        "svip": "3000",
    }

    def test_capacity_all_present(self):
        rows = _load_sys_dict_rows()
        caps = {name: value for tc, name, value in rows if tc == "favorite_capacity"}
        assert len(caps) == 4, f"favorite_capacity 应为 4 项，实际 {len(caps)}"
        for key, value in self.CAPACITY_DEFAULTS.items():
            assert key in caps, f"缺失收藏容量 {key}"
            assert caps[key] == value, f"{key} 种子值 {caps[key]} 与默认 {value} 不一致"

    def test_level_mapping_and_fallback_match_seeds(self):
        from app.service.favorite_service import CAPACITY_DEFAULTS, LEVEL_TO_CAPACITY_KEY

        assert LEVEL_TO_CAPACITY_KEY == {
            "level_0": "default",
            "level_1": "vip1",
            "level_2": "vip2",
            "level_3": "svip",
        }
        for key, seed_value in self.CAPACITY_DEFAULTS.items():
            assert str(CAPACITY_DEFAULTS[key]) == seed_value, (
                f"{key} 消费方回退默认 {CAPACITY_DEFAULTS[key]} 与种子 {seed_value} 不一致"
            )
