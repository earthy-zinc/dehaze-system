package dataset

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
)

// ImageInfo 图片信息
type ImageInfo struct {
	Width  int
	Height int
	Format string
	Size   int64
}

// PairedImageValidator 配对图片校验器
// 负责校验清晰图和雾霾图的分辨率是否一致
type PairedImageValidator struct{}

// NewPairedImageValidator 创建配对图片校验器
func NewPairedImageValidator() *PairedImageValidator {
	return &PairedImageValidator{}
}

// ValidateResolution 校验配对图片的分辨率是否一致
// clearImageFile: 清晰图文件
// hazyImageFiles: 雾霾图文件列表
func (v *PairedImageValidator) ValidateResolution(clearImageFile string, hazyImageFiles []string) error {
	// 获取清晰图的分辨率
	clearInfo, err := v.GetImageInfo(clearImageFile)
	if err != nil {
		return fmt.Errorf("无法读取清晰图 %s: %w", clearImageFile, err)
	}

	// 校验每张雾霾图的分辨率
	for _, hazyFile := range hazyImageFiles {
		hazyInfo, err := v.GetImageInfo(hazyFile)
		if err != nil {
			return fmt.Errorf("无法读取雾霾图 %s: %w", hazyFile, err)
		}

		if hazyInfo.Width != clearInfo.Width || hazyInfo.Height != clearInfo.Height {
			return fmt.Errorf(
				"分辨率不一致: %s(%dx%d) vs %s(%dx%d)",
				filepath.Base(clearImageFile), clearInfo.Width, clearInfo.Height,
				filepath.Base(hazyFile), hazyInfo.Width, hazyInfo.Height,
			)
		}
	}

	return nil
}

// GetImageInfo 获取图片文件信息
func (v *PairedImageValidator) GetImageInfo(filePath string) (*ImageInfo, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// 获取文件大小
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, err
	}

	// 解码图片配置
	config, _, err := image.DecodeConfig(file)
	if err != nil {
		return nil, fmt.Errorf("图片格式不支持或已损坏: %w", err)
	}

	// 获取图片格式
	ext := strings.ToLower(filepath.Ext(filePath))

	return &ImageInfo{
		Width:  config.Width,
		Height: config.Height,
		Format: ext,
		Size:   fileInfo.Size(),
	}, nil
}

// 注：原 ValidateHazeLevel / ValidateHazyImageCount / ValidatePairedImages 方法已移除。
// 原因：haze_level 字段支持多种规范（light/medium/heavy、beta=X、A=X,beta=Y 等），
// 不再做硬性枚举校验。清晰图和有雾图均为可选（适配不同数据集规范）。
// 详见需求规格.md 2.6.2 节和后端实现.md 4.2 节。
