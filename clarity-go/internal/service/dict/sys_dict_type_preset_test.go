package dict

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// expectedSystemPresetDictTypeCodes 预置字典类型码的期望集合。
//
// 事实源：python app/service/dict_service.py 的 SYSTEM_PRESET_DICT_TYPE_CODES。
// 该集合决定 T-DM-025「系统预置字典类型不可删除」的保护范围；任一端少一项，
// 就会允许删除事实源禁止删除的字典类型（曾出现 java/go 缺 ai_model → 可删掉
// "平台默认模型" 字典类型，而 python 禁止），造成三端行为分裂。
var expectedSystemPresetDictTypeCodes = map[string]bool{
	"gender":                true,
	"ai_guardrail_defaults": true,
	"ai_provider_health":    true,
	"ai_embedding":          true,
	"ai_model":              true,
	"member_growth_rules":   true,
	"favorite_capacity":     true,
	"ai_eval":               true,
}

// TestSystemPresetDictTypeCodesMatchPython 预置码集合必须与 python 事实源逐项相等（顺序无关）。
// 增删本集合任一项都会让本用例失败，用作跨端一致性护栏。
func TestSystemPresetDictTypeCodesMatchPython(t *testing.T) {
	require.Equal(t, expectedSystemPresetDictTypeCodes, systemPresetDictTypeCodes,
		"预置字典类型码集合必须与 python SYSTEM_PRESET_DICT_TYPE_CODES 完全一致（顺序无关）")
}
