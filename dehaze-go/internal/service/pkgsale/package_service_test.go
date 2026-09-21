package pkgsale

import (
	"encoding/json"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// benefitOverridesCanonical 是权益覆盖项的权威 key 集合：python `service/package_service.py`
// 的 `BENEFIT_FIELDS` 与 java `model/form/BenefitOverrides` 的字段集完全一致（17 项）。
// go 的 `bo.BenefitOverrides` 少一个字段，该 key 就会在保存时被静默丢弃（json 未知字段不报错），
// 而履约侧（`member_service` 的 aiCategory 覆盖、`order_service` 的配额覆盖）仍会按 key 读取它。
var benefitOverridesCanonical = map[string]int{
	"monthlyDehazeQuota":          1,
	"monthlyDerainQuota":          2,
	"monthlyDesnowQuota":          3,
	"monthlyLowlightQuota":        4,
	"monthlySuperResolutionQuota": 5,
	"monthlyDenoiseQuota":         6,
	"monthlyInpaintQuota":         7,
	"monthlyEvaluateQuota":        8,
	"aiCreditsDaily":              9,
	"aiCreditsMonthly":            10,
	"historyRetention":            11,
	"batchLimit":                  12,
	"priority":                    13,
	"advancedParams":              14,
	"hdExport":                    15,
	"reportExport":                16,
	"batchDownload":               17,
}

// TestBenefitOverridesRoundTripKeepsEveryKey 覆盖套餐 CRUD 的"读回 → 保存"往返：
// 读回走 `PackageService.GetForm` 的 `json.Unmarshal(..., &bo.BenefitOverrides)`（:361-362），
// 保存走 `Create`/`Update` 的 `json.Marshal(form.BenefitOverrides)`（:438、:496）——
// 两次都经过该结构体，故结构体缺字段即为端到端数据丢失。
//
// 缺字段时本用例会以"map 少 key / 键值为 null"失败；曾缺 aiCreditsDaily、aiCreditsMonthly
// 与 6 个 monthly*Quota（由 java 写入或种子数据落库的 key，经 go 保存后被抹掉）。
func TestBenefitOverridesRoundTripKeepsEveryKey(t *testing.T) {
	stored, err := json.Marshal(benefitOverridesCanonical)
	require.NoError(t, err)

	// 读回：GetForm 路径
	var overrides bo.BenefitOverrides
	require.NoError(t, json.Unmarshal(stored, &overrides))

	// 保存：Create/Update 路径
	data, err := json.Marshal(&overrides)
	require.NoError(t, err)

	var saved map[string]int
	require.NoError(t, json.Unmarshal(data, &saved))
	assert.Equal(t, benefitOverridesCanonical, saved,
		"套餐保存经 bo.BenefitOverrides 序列化，任何字段缺失都会静默抹掉该 key")
}
