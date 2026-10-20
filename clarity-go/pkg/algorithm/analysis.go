package algorithm

import (
	"crypto/md5"
	"encoding/hex"
	"math"
	"math/big"
)

// AnalyzeImageResponse 图像特征分析响应（字段与 python ImageFeatureAnalysisVO 逐项对应）
type AnalyzeImageResponse struct {
	ImageMd5          string       `json:"imageMd5"`
	HazeLevel         string       `json:"hazeLevel"`
	HazeConfidence    float64      `json:"hazeConfidence"`
	SceneType         string       `json:"sceneType"`
	SceneConfidence   float64      `json:"sceneConfidence"`
	Lighting          string       `json:"lighting"`
	Complexity        float64      `json:"complexity"`
	ColorDistribution ColorFeature `json:"colorDistribution"`
	Resolution        string       `json:"resolution"`
	NoiseLevel        string       `json:"noiseLevel"`
}

// ColorFeature 颜色分布特征
type ColorFeature struct {
	Temperature float64 `json:"temperature"`
	Saturation  float64 `json:"saturation"`
}

// 与 python recommendation_service 逐项一致的取值表（顺序即取模结果的语义，勿调整）。
var (
	validHazeLevels  = []string{"light", "moderate", "heavy"}
	validSceneTypes  = []string{"urban", "landscape", "building", "night", "backlight", "indoor"}
	validLightings   = []string{"bright", "normal", "dark", "veryDark", "backlight"}
	validResolutions = []string{"sd", "hd", "uhd"}
	validNoiseLevels = []string{"low", "medium", "high"}
)

// AnalyzeImage 本地复刻 python `recommendation_service.analyze` 的确定性特征算法：
// 以 URL 的 MD5 作固定种子（`seed = int(md5, 16)`），从固定表取模派生特征，**零外部依赖**。
//
// 之所以本地复刻而非委托 python `/api/v1/recommendations/analyze`：该算法是纯确定性派生、可逐位对齐，
// 本地实现省掉一跳网络与鉴权面。
// （注：早前此处写"内部服务未带认证会得到 A0230、形成鉴权死结"是误判——A0230 的真实原因是运行库缺
// M2M 种子行，补种后内部调用即通。推理类端点依赖真实模型权重，仍必须委托 python，不可本地复刻。）
//
// 注意：python 的 seed 是 **128 位整数**（无溢出概念），Go 的 int64 装不下，故全程用 math/big。
// 同一 URL 必须与 python 得到完全相同的 md5 与特征（已由 analysis_test.go 的黄金值钉住）。
func AnalyzeImage(imageURL string) *AnalyzeImageResponse {
	sum := md5.Sum([]byte(imageURL))
	md5Hex := hex.EncodeToString(sum[:])
	seed, _ := new(big.Int).SetString(md5Hex, 16) // md5 十六进制串必然合法，不会失败

	return &AnalyzeImageResponse{
		ImageMd5:        md5Hex,
		HazeLevel:       validHazeLevels[mod(seed, len(validHazeLevels))],
		HazeConfidence:  round2(0.5 + float64(mod(seed, 50))/100.0),
		SceneType:       validSceneTypes[mod(seed, len(validSceneTypes))],
		SceneConfidence: round2(0.5 + float64(divMod(seed, 7, 50))/100.0),
		Lighting:        validLightings[mod(seed, len(validLightings))],
		Complexity:      round2(0.3 + float64(divMod(seed, 11, 70))/100.0),
		ColorDistribution: ColorFeature{
			Temperature: round2(4000.0 + float64(mod(seed, 6000))),
			Saturation:  round2(0.3 + float64(divMod(seed, 13, 70))/100.0),
		},
		Resolution: validResolutions[mod(seed, len(validResolutions))],
		NoiseLevel: validNoiseLevels[mod(seed, len(validNoiseLevels))],
	}
}

// mod 复刻 python `seed % n`。
func mod(seed *big.Int, n int) int {
	return int(new(big.Int).Mod(seed, big.NewInt(int64(n))).Int64())
}

// divMod 复刻 python `(seed // d) % m`。
func divMod(seed *big.Int, d, m int) int {
	quotient := new(big.Int).Div(seed, big.NewInt(int64(d)))
	return int(new(big.Int).Mod(quotient, big.NewInt(int64(m))).Int64())
}

// round2 复刻 python `round(x, 2)`：本算法的取值本身即两位小数，不存在半值舍入歧义。
func round2(value float64) float64 {
	return math.Round(value*100) / 100
}
