package aidomain

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDiffSnapshotNestedAndList(t *testing.T) {
	base := map[string]any{
		"name": "旧名称",
		"config": map[string]any{
			"max_steps": 10.0,
			"guardrails": map[string]any{
				"pii_mask": map[string]any{"enabled": true},
			},
		},
		"skills": []any{"a", "b"},
	}
	target := map[string]any{
		"name": "新名称",
		"config": map[string]any{
			"max_steps": 20.0,
			"guardrails": map[string]any{
				"pii_mask": map[string]any{"enabled": false},
			},
		},
		"skills": []any{"a", "b", "c"},
	}
	diffs := []map[string]any{}
	diffSnapshot(base, target, "", &diffs)

	fields := map[string]map[string]any{}
	for _, diff := range diffs {
		fields[diff["field"].(string)] = diff
	}
	require.Contains(t, fields, "name")
	assert.Equal(t, "旧名称", fields["name"]["base"])
	assert.Equal(t, "新名称", fields["name"]["target"])
	require.Contains(t, fields, "config.max_steps")
	require.Contains(t, fields, "config.guardrails.pii_mask.enabled")
	// 列表整体比较，不逐元素展开
	require.Contains(t, fields, "skills")
}

func TestDiffSnapshotIdentical(t *testing.T) {
	snapshot := map[string]any{"name": "同一", "skills": []any{"a"}}
	diffs := []map[string]any{}
	diffSnapshot(snapshot, snapshot, "", &diffs)
	assert.Empty(t, diffs)
}

func TestSnapshotSubagents(t *testing.T) {
	endpointID := 9.0
	snapshot := map[string]any{
		"subagents": []any{
			map[string]any{"agent_id": 3.0, "priority": 1.0, "endpoint_id": nil},
			map[string]any{"agent_id": 4.0, "priority": 2.0, "endpoint_id": endpointID},
		},
	}
	links := snapshotSubagents(snapshot)
	require.Len(t, links, 2)
	assert.Equal(t, int64(3), links[0].SubagentAgentID)
	assert.Nil(t, links[0].EndpointID)
	assert.Equal(t, int64(4), links[1].SubagentAgentID)
	require.NotNil(t, links[1].EndpointID)
	assert.Equal(t, int64(9), *links[1].EndpointID)
}

func TestSnapshotStrings(t *testing.T) {
	snapshot := map[string]any{"skills": []any{"s1", "s2", 3.0}}
	assert.Equal(t, []string{"s1", "s2"}, snapshotStrings(snapshot, "skills"))
	assert.Equal(t, []string{}, snapshotStrings(map[string]any{}, "missing"))
}

func TestSplitDottedAndCoerceScalar(t *testing.T) {
	assert.Equal(t, []string{"prompt_injection", "enabled"}, splitDotted("prompt_injection.enabled"))
	assert.Equal(t, []string{"enabled"}, splitDotted("enabled"))
	assert.Equal(t, true, coerceScalar("true"))
	assert.Equal(t, false, coerceScalar("false"))
	assert.Equal(t, 30, coerceScalar("30"))
	assert.Equal(t, 0.8, coerceScalar("0.8"))
	assert.Equal(t, "abc", coerceScalar("abc"))
}

func TestNormalizeConfigJSONDropsNil(t *testing.T) {
	raw := json.RawMessage(`{"max_steps":20,"tool_timeout":null,"guardrails":{"pii_mask":{"enabled":true},"fact_check":null}}`)
	normalized := normalizeConfigJSON(raw)
	var parsed map[string]any
	require.NoError(t, json.Unmarshal([]byte(normalized), &parsed))
	assert.Equal(t, float64(20), parsed["max_steps"])
	_, hasTimeout := parsed["tool_timeout"]
	assert.False(t, hasTimeout)
	guardrails := parsed["guardrails"].(map[string]any)
	_, hasFactCheck := guardrails["fact_check"]
	assert.False(t, hasFactCheck)
	require.Contains(t, guardrails, "pii_mask")
}

func TestApplySnapshotString(t *testing.T) {
	fields := map[string]any{"update_by": int64(1)}
	snapshot := map[string]any{"name": "名称", "system_prompt": nil}
	applySnapshotString(fields, snapshot, "name", "name")
	applySnapshotString(fields, snapshot, "system_prompt", "system_prompt")
	applySnapshotString(fields, snapshot, "missing", "model_id")
	assert.Equal(t, "名称", fields["name"])
	assert.Nil(t, fields["system_prompt"])
	_, hasModelID := fields["model_id"]
	assert.False(t, hasModelID)
}

// TestReasoningDefaultsContractMatchesConstant 断言对外契约与代码常量同值同键：
// 常量（reasoningDefaults）是唯一事实源，契约映射写错或字段改名必须在此失败。
func TestReasoningDefaultsContractMatchesConstant(t *testing.T) {
	defaults := ReasoningDefaults()
	assert.Equal(t, reasoningDefaults["max_steps_react"], defaults.MaxStepsReact)
	assert.Equal(t, reasoningDefaults["max_steps_plan"], defaults.MaxStepsPlan)
	assert.Equal(t, reasoningDefaults["max_steps_reflexion"], defaults.MaxStepsReflexion)
	assert.Equal(t, reasoningDefaults["max_iterations_reflexion"], defaults.MaxIterationsReflexion)
	assert.Equal(t, reasoningDefaults["reflexion_threshold"], defaults.ReflexionThreshold)
	assert.Equal(t, reasoningDefaults["max_parallel"], defaults.MaxParallel)
	assert.Equal(t, reasoningDefaults["tool_timeout"], defaults.ToolTimeout)
	assert.Equal(t, reasoningDefaults["token_budget"], defaults.TokenBudget)
	assert.Equal(t, reasoningDefaults["retry_max"], defaults.RetryMax)

	// 对外 JSON 键名固定为 camelCase（对齐 python AgentConfigDefaults / java AiAgentConfigDefaultsVO）
	raw, err := json.Marshal(defaults)
	require.NoError(t, err)
	fields := map[string]any{}
	require.NoError(t, json.Unmarshal(raw, &fields))
	assert.Len(t, fields, 9)
	for _, key := range []string{
		"maxStepsReact", "maxStepsPlan", "maxStepsReflexion", "maxIterationsReflexion",
		"reflexionThreshold", "maxParallel", "toolTimeout", "tokenBudget", "retryMax",
	} {
		assert.Contains(t, fields, key)
	}
}
