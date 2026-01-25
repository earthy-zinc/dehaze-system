package utils

import (
	"sort"
	"strings"
)

// HazeLevel 雾霾程度枚举
type HazeLevel string

const (
	HazeLevelLight  HazeLevel = "light"  // 轻度雾霾
	HazeLevelMedium HazeLevel = "medium" // 中度雾霾
	HazeLevelHeavy  HazeLevel = "heavy"  // 重度雾霾
)

// HazeLevelOrder 雾霾程度排序值
func (h HazeLevel) Order() int {
	switch strings.ToLower(string(h)) {
	case string(HazeLevelLight):
		return 1
	case string(HazeLevelMedium):
		return 2
	case string(HazeLevelHeavy):
		return 3
	default:
		return 99
	}
}

// ImageClassificationResult 图片分类结果
type ImageClassificationResult struct {
	ClearImage *ClassifiedImage // 清晰图
	HazyImages ClassifiedImages // 有雾图列表
	SceneType  string           // 场景类型
}

// ClassifiedImage 分类后的图片信息
type ClassifiedImage struct {
	ID        int64     `json:"id"`
	Type      string    `json:"type"`
	URL       string    `json:"url"`
	OriginURL string    `json:"originUrl"`
	HazeLevel HazeLevel `json:"hazeLevel"`
	SceneType string    `json:"sceneType"`
	Name      string    `json:"name"`
}

// ClassifiedImages 分类后的图片列表
type ClassifiedImages []*ClassifiedImage

// ImageClassificationUtils 图片分类工具类
type ImageClassificationUtils struct{}

// NewImageClassificationUtils 创建图片分类工具实例
func NewImageClassificationUtils() *ImageClassificationUtils {
	return &ImageClassificationUtils{}
}

// IsClearImage 判断类型字符串是否表示清晰图
// 支持：clear、clean、清晰、无雾等关键词
func (u *ImageClassificationUtils) IsClearImage(imageType string) bool {
	if imageType == "" {
		return false
	}
	lowerType := strings.ToLower(imageType)
	return strings.Contains(lowerType, "clear") ||
		strings.Contains(lowerType, "clean") ||
		strings.Contains(imageType, "清晰") ||
		strings.Contains(imageType, "无雾")
}

// IsHazyImage 判断类型字符串是否表示有雾图
// 支持：haze、hazy、有雾等关键词
func (u *ImageClassificationUtils) IsHazyImage(imageType string) bool {
	if imageType == "" {
		return false
	}
	lowerType := strings.ToLower(imageType)
	return strings.Contains(lowerType, "haze") ||
		strings.Contains(lowerType, "hazy") ||
		strings.Contains(imageType, "有雾")
}

// ClassifyImages 对图片列表进行分类
// 返回：清晰图/有雾图分离、按hazeLevel排序、sceneType提取
func (u *ImageClassificationUtils) ClassifyImages(images []ClassifiedImage) *ImageClassificationResult {
	result := &ImageClassificationResult{
		HazyImages: ClassifiedImages{},
	}

	if len(images) == 0 {
		return result
	}

	var clearImage *ClassifiedImage
	var hazyImages ClassifiedImages

	for i := range images {
		image := &images[i]
		if u.IsClearImage(image.Type) {
			if clearImage == nil {
				clearImage = &ClassifiedImage{
					ID:        image.ID,
					Type:      image.Type,
					URL:       image.URL,
					OriginURL: image.OriginURL,
					HazeLevel: image.HazeLevel,
					SceneType: image.SceneType,
					Name:      image.Name,
				}
			}
		} else if u.IsHazyImage(image.Type) {
			hazyImages = append(hazyImages, &ClassifiedImage{
				ID:        image.ID,
				Type:      image.Type,
				URL:       image.URL,
				OriginURL: image.OriginURL,
				HazeLevel: image.HazeLevel,
				SceneType: image.SceneType,
				Name:      image.Name,
			})
		}
	}

	// 按雾霾程度排序有雾图
	u.SortByHazeLevel(hazyImages)

	result.ClearImage = clearImage
	result.HazyImages = hazyImages

	// 提取场景类型
	result.SceneType = u.ExtractSceneType(clearImage, hazyImages)

	return result
}

// SortByHazeLevel 按雾霾程度排序
// 排序规则：按严重程度排序（light < medium < heavy），空值排最后
func (u *ImageClassificationUtils) SortByHazeLevel(images ClassifiedImages) {
	if len(images) <= 1 {
		return
	}

	sort.Slice(images, func(i, j int) bool {
		orderI := images[i].HazeLevel.Order()
		orderJ := images[j].HazeLevel.Order()
		return orderI < orderJ
	})
}

// ExtractSceneType 提取场景类型
// 优先从清晰图获取，否则从第一张有雾图获取
func (u *ImageClassificationUtils) ExtractSceneType(clearImage *ClassifiedImage, hazyImages ClassifiedImages) string {
	// 优先从清晰图获取
	if clearImage != nil && clearImage.SceneType != "" {
		return clearImage.SceneType
	}
	// 其次从第一张有雾图获取
	if len(hazyImages) > 0 && hazyImages[0] != nil && hazyImages[0].SceneType != "" {
		return hazyImages[0].SceneType
	}
	return ""
}
