package algorithm

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestAnalyzeImageMatchesPythonDeterministic 与 python `recommendation_service.analyze` 逐字段对齐。
//
// 黄金值由 python 直接产出（`hashlib.md5(url)` → `int(md5,16)` 后按同一套取模派生），
// 故本用例是**跨端事实源对照**而非自证：python 换算法或 Go 复刻走样都会在此失败。
// 覆盖三种输入形态：普通 URL、带 query 的 URL（校验时剥离 `?` 后缀但 md5 用原串）、http 协议。
func TestAnalyzeImageMatchesPythonDeterministic(t *testing.T) {
	cases := []struct {
		url  string
		want AnalyzeImageResponse
	}{
		{
			url: "https://cdn.example.com/haze/1.jpg",
			want: AnalyzeImageResponse{
				ImageMd5:        "304e67adcc157ee3477194981e8eab0c",
				HazeLevel:       "moderate",
				HazeConfidence:  0.64,
				SceneType:       "backlight",
				SceneConfidence: 0.52,
				Lighting:        "backlight",
				Complexity:      0.75,
				ColorDistribution: ColorFeature{
					Temperature: 4364,
					Saturation:  0.52,
				},
				Resolution: "hd",
				NoiseLevel: "medium",
			},
		},
		{
			// 带 query：python `_resolve_and_validate_image_url` 只用剥离 `?` 后的串做后缀校验，
			// 但返回原始 URL，md5 因此包含 query —— 此处钉住该细节。
			url: "https://cdn.example.com/a/b?x=1.png",
			want: AnalyzeImageResponse{
				ImageMd5:        "71f91f669ecfd37d1acc9fee2b1fd962",
				HazeLevel:       "moderate",
				HazeConfidence:  0.92,
				SceneType:       "backlight",
				SceneConfidence: 0.7,
				Lighting:        "dark",
				Complexity:      0.74,
				ColorDistribution: ColorFeature{
					Temperature: 7442,
					Saturation:  0.51,
				},
				Resolution: "hd",
				NoiseLevel: "medium",
			},
		},
		{
			url: "http://x/y.webp",
			want: AnalyzeImageResponse{
				ImageMd5:        "60255a32739eaf0bd2976f33ccacb374",
				HazeLevel:       "light",
				HazeConfidence:  0.56,
				SceneType:       "urban",
				SceneConfidence: 0.93,
				Lighting:        "normal",
				Complexity:      0.57,
				ColorDistribution: ColorFeature{
					Temperature: 5956,
					Saturation:  0.48,
				},
				Resolution: "sd",
				NoiseLevel: "low",
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.url, func(t *testing.T) {
			assert.Equal(t, tc.want, *AnalyzeImage(tc.url))
		})
	}
}

// TestAnalyzeImageIsStableAcrossCalls 同 URL 结果必须稳定（python 以 MD5 为固定种子，
// 跨进程/跨重启一致；这里钉住 Go 侧无随机、无时间因素）。
func TestAnalyzeImageIsStableAcrossCalls(t *testing.T) {
	url := "http://x/y.webp" // 黄金值见上表
	first := AnalyzeImage(url)
	assert.Equal(t, "60255a32739eaf0bd2976f33ccacb374", first.ImageMd5)
	for i := 0; i < 5; i++ {
		assert.Equal(t, *first, *AnalyzeImage(url))
	}
}
