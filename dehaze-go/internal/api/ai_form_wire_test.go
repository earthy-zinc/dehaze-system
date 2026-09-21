package api

import (
	"encoding/json"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	aidomain "github.com/earthyzinc/dehaze-go/internal/service/aidomain"
	"github.com/go-playground/validator/v10"
	"github.com/stretchr/testify/require"
)

// TestStatusFormAllowsZero 回归：启停表单的 `status=0`（禁用）是合法值。
// `Status int8 \`binding:"required,oneof=0 1"\“ 会被 validator 判为"缺失"→ A0400（SDK 启停用例全红），
// 故改用指针承接 required；此处用与 gin 相同的 tag 名复现该判定。
func TestStatusFormAllowsZero(t *testing.T) {
	v := validator.New()
	v.SetTagName("binding")
	zero, two := int8(0), int8(2)

	require.NoError(t, v.Struct(&bo.SkillStatusForm{Status: &zero}), "status=0 必须通过")
	require.NoError(t, v.Struct(&bo.McpServerStatusForm{Status: &zero}), "status=0 必须通过")
	require.Error(t, v.Struct(&bo.SkillStatusForm{}), "status 缺失必须被拒")
	require.Error(t, v.Struct(&bo.SkillStatusForm{Status: &two}), "status=2 必须被拒")
}

// TestFormsBindSnakeCaseWire 回归：这些表单在 python 侧是**纯 BaseModel（无 camelCase 别名）**，
// wire 字段为 snake_case；go 曾用 camelCase 标签 → 字段读成零值（评测集「类型取值非法」、
// 子 Agent「不存在: 0」、复制 Agent「编码不能为空」）。此处直接钉 wire 字段名。
func TestFormsBindSnakeCaseWire(t *testing.T) {
	t.Run("评测集创建 dataset_type", func(t *testing.T) {
		var form aidomain.EvalDatasetCreateForm
		require.NoError(t, json.Unmarshal([]byte(`{"name":"n","description":"d","dataset_type":"dev"}`), &form))
		require.Equal(t, "dev", form.DatasetType)
	})

	t.Run("评测样本 snake_case 全字段", func(t *testing.T) {
		body := `{"task_goal":"g","allowed_input":"text","tools":["file_read"],
			"expected_process":"p","expected_result":"r","forbidden_behavior":"f","risk_level":"high"}`
		var form aidomain.EvalSampleCreateForm
		require.NoError(t, json.Unmarshal([]byte(body), &form))
		require.Equal(t, "g", form.TaskGoal)
		require.NotNil(t, form.AllowedInput)
		require.Equal(t, "p", *form.ExpectedProcess)
		require.Equal(t, "r", *form.ExpectedResult)
		require.Equal(t, "f", *form.ForbiddenBehavior)
		require.Equal(t, "high", form.RiskLevel)
		require.Equal(t, []string{"file_read"}, form.Tools)
	})

	t.Run("子 Agent 关联 agent_id/endpoint_id", func(t *testing.T) {
		var form aidomain.AgentSubAgentsForm
		body := `{"subagents":[{"agent_id":80,"endpoint_id":null,"priority":1}]}`
		require.NoError(t, json.Unmarshal([]byte(body), &form))
		require.Len(t, form.Subagents, 1)
		require.EqualValues(t, 80, form.Subagents[0].AgentID)
		require.Nil(t, form.Subagents[0].EndpointID)
		require.Equal(t, 1, form.Subagents[0].Priority)
	})

	t.Run("Agent MCP 关联 mcp_namespaces", func(t *testing.T) {
		var form aidomain.AgentMcpForm
		require.NoError(t, json.Unmarshal([]byte(`{"mcp_namespaces":["ns_a"]}`), &form))
		require.Equal(t, []string{"ns_a"}, form.McpNamespaces)
	})
}
