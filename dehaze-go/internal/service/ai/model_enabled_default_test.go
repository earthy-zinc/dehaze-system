package ai

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	dictservice "github.com/earthyzinc/dehaze-go/internal/service/dict"
	"github.com/stretchr/testify/require"
)

// stubDictService 只实现 GetByTypeCode（其余方法由嵌入的接口占位，测试不会触达）。
type stubDictService struct {
	dictservice.IDictService

	options []vo.Option
	err     error
}

func (s stubDictService) GetByTypeCode(_ context.Context, _ string) ([]vo.Option, error) {
	return s.options, s.err
}

// TestPlatformDefaultDictKey 平台默认模型字典键口径（三端同源）：
// chat 用 `default_model`，其余 `default_model.{modelType}`。
func TestPlatformDefaultDictKey(t *testing.T) {
	require.Equal(t, "default_model", platformDefaultDictKey("chat"))
	require.Equal(t, "default_model.image_edit", platformDefaultDictKey("image_edit"))
	require.Equal(t, "default_model.asr", platformDefaultDictKey("asr"))
	require.Equal(t, "default_model.tts", platformDefaultDictKey("tts"))
	require.Equal(t, "ai_model", platformDefaultDictType)
}

// TestMarkPlatformDefaults 启用模型列表的 isDefault 标记：取值与字典一致
// （按 model_type 分键命中平台默认的模型为 true，其余 false）。
func TestMarkPlatformDefaults(t *testing.T) {
	svc := &ModelService{dictSvc: stubDictService{options: []vo.Option{
		{Value: "chat-a", Label: "default_model"},
		{Value: "aecrnet", Label: "default_model.image_edit"},
	}}}

	items := []vo.AiModelVO{
		{ModelID: "chat-a", ModelType: "chat"},
		{ModelID: "chat-b", ModelType: "chat"},
		{ModelID: "aecrnet", ModelType: "image_edit"},
		{ModelID: "no-default-type", ModelType: "video_gen"},
	}
	svc.markPlatformDefaults(context.Background(), items)

	flags := map[string]bool{}
	for _, item := range items {
		require.NotNil(t, item.IsDefault, "启用列表每项都必须带标记（用于前端预选后端默认模型）")
		flags[item.ModelID] = *item.IsDefault
	}
	require.Equal(t, map[string]bool{
		"chat-a":          true,
		"chat-b":          false,
		"aecrnet":         true,
		"no-default-type": false,
	}, flags)
}

// TestMarkPlatformDefaultsSameModelAcrossProviders 同一 model_id 配多供应商时该模型各行都为 true
// （标记语义＝"该模型是否为其类型的平台默认模型"）。
func TestMarkPlatformDefaultsSameModelAcrossProviders(t *testing.T) {
	svc := &ModelService{dictSvc: stubDictService{options: []vo.Option{
		{Value: "aecrnet", Label: "default_model.image_edit"},
	}}}

	items := []vo.AiModelVO{
		{ModelID: "aecrnet", ModelType: "image_edit", ProviderID: 1},
		{ModelID: "aecrnet", ModelType: "image_edit", ProviderID: 2},
	}
	svc.markPlatformDefaults(context.Background(), items)

	require.True(t, *items[0].IsDefault)
	require.True(t, *items[1].IsDefault)
}

// TestMarkPlatformDefaultsUnconfigured 字典未配置/停用/读取失败时全部 false，不回落代码常量、
// 也不回落到"列表首项"。
func TestMarkPlatformDefaultsUnconfigured(t *testing.T) {
	cases := map[string]*ModelService{
		"未注入字典服务": {dictSvc: nil},
		"字典无该键":   {dictSvc: stubDictService{options: []vo.Option{{Value: "x", Label: "other_key"}}}},
		"字典值为空":   {dictSvc: stubDictService{options: []vo.Option{{Value: "", Label: "default_model"}}}},
		"读取失败":    {dictSvc: stubDictService{err: errors.New("dict unavailable")}},
	}
	for name, svc := range cases {
		t.Run(name, func(t *testing.T) {
			items := []vo.AiModelVO{{ModelID: "chat-a", ModelType: "chat"}}
			svc.markPlatformDefaults(context.Background(), items)
			require.NotNil(t, items[0].IsDefault)
			require.False(t, *items[0].IsDefault)
		})
	}
}

// TestAiModelVOIsDefaultJSONKey 对外契约键名：isDefault（camelCase），未计算时省略（omitempty）。
func TestAiModelVOIsDefaultJSONKey(t *testing.T) {
	flag := true
	raw, err := json.Marshal(vo.AiModelVO{ModelID: "aecrnet", ModelType: "image_edit", IsDefault: &flag})
	require.NoError(t, err)
	require.Contains(t, string(raw), `"isDefault":true`)

	raw, err = json.Marshal(vo.AiModelVO{ModelID: "aecrnet", ModelType: "image_edit"})
	require.NoError(t, err)
	require.NotContains(t, string(raw), "isDefault")
}
