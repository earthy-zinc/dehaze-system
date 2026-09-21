package ai

import (
	"context"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/stretchr/testify/require"
)

// TestCreateSkillDefaultsDisabled 回归（SDK ai-skill.test.ts T-MF-082「新建 Skill 默认禁用 status=0」）：
// model 的 Status 一旦带 `default:1`，GORM 会把零值替换成默认值并省略该列 → status=0 永远写不进去，
// 新建 Skill 变成"已启用"。python 侧创建时显式 `status=_STATUS_DISABLED`，故 go 必须能插 0。
func TestCreateSkillDefaultsDisabled(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := NewSkillService(airepo.NewSkillRepository(db), nil)

	created, err := svc.CreateSkill(context.Background(), &bo.SkillCreateForm{
		Name:        "test_skill_default_status_probe",
		Description: "默认禁用回归",
		Scene:       "通用",
		Instruction: "# 测试 Skill 指令\n按步骤执行测试流程",
	}, 900001)
	require.NoError(t, err)
	require.EqualValues(t, 0, created.Status, "新建 Skill 必须默认禁用（status=0）")

	var stored int8
	require.NoError(t, db.Raw("SELECT status FROM sys_ai_skill WHERE id = ?", created.ID).
		Row().Scan(&stored))
	require.EqualValues(t, 0, stored, "落库值必须为 0，不得被 GORM 默认值替换成 1")
}
