package dataset

import (
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
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

// ValidateResolutionFromReaders 从读取器校验配对图片的分辨率
func (v *PairedImageValidator) ValidateResolutionFromReaders(clearReader io.Reader, hazyReaders []io.Reader) error {
	// 获取清晰图的分辨率
	clearInfo, err := v.GetImageInfoFromReader(clearReader)
	if err != nil {
		return fmt.Errorf("无法读取清晰图: %w", err)
	}

	// 校验每张雾霾图的分辨率
	for i, reader := range hazyReaders {
		hazyInfo, err := v.GetImageInfoFromReader(reader)
		if err != nil {
			return fmt.Errorf("无法读取第%d张雾霾图: %w", i+1, err)
		}

		if hazyInfo.Width != clearInfo.Width || hazyInfo.Height != clearInfo.Height {
			return fmt.Errorf(
				"分辨率不一致: 清晰图(%dx%d) vs 第%d张雾霾图(%dx%d)",
				clearInfo.Width, clearInfo.Height, i+1, hazyInfo.Width, hazyInfo.Height,
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

// GetImageInfoFromReader 从读取器获取图片信息
func (v *PairedImageValidator) GetImageInfoFromReader(reader io.Reader) (*ImageInfo, error) {
	config, _, err := image.DecodeConfig(reader)
	if err != nil {
		return nil, fmt.Errorf("图片格式不支持或已损坏: %w", err)
	}

	return &ImageInfo{
		Width:  config.Width,
		Height: config.Height,
		Format: "unknown",
		Size:   0,
	}, nil
}

// allowedFormats 文档规定的允许格式：jpg/jpeg/png/gif
var allowedFormats = map[string]bool{
	".jpg":  true,
	".jpeg": true,
	".png":  true,
	".gif":  true,
}

// validHazeLevels 文档规定的有效雾霾程度
var validHazeLevels = map[string]bool{
	"light":  true,
	"medium": true,
	"heavy":  true,
}

// ValidateImageType 校验图片类型是否支持（仅允许 jpg/jpeg/png/gif）
func (v *PairedImageValidator) ValidateImageType(filename string) error {
	ext := strings.ToLower(filepath.Ext(filename))

	if !allowedFormats[ext] {
		return fmt.Errorf("不支持的图片格式: %s，仅支持 jpg/jpeg/png/gif", ext)
	}

	return nil
}

// ValidateHazeLevel 校验雾霾程度是否有效（仅允许 light/medium/heavy）
func (v *PairedImageValidator) ValidateHazeLevel(hazeLevel string) error {
	if !validHazeLevels[hazeLevel] {
		return fmt.Errorf("无效的雾霾程度: %s，仅支持 light/medium/heavy", hazeLevel)
	}
	return nil
}

// ValidateHazyImageCount 校验有雾图数量与雾霾程度数量是否一致
func (v *PairedImageValidator) ValidateHazyImageCount(hazyCount int, hazeLevels []string) error {
	if hazyCount != len(hazeLevels) {
		return fmt.Errorf("有雾图数量(%d)与雾霾程度数量(%d)不一致", hazyCount, len(hazeLevels))
	}
	for i, level := range hazeLevels {
		if err := v.ValidateHazeLevel(level); err != nil {
			return fmt.Errorf("第%d张有雾图: %w", i+1, err)
		}
	}
	return nil
}

// ValidatePairedImages 完整的配对图片校验（清晰图必填、有雾图必填、数量匹配、雾霾程度有效）
func (v *PairedImageValidator) ValidatePairedImages(hasClearImage bool, hazyCount int, hazeLevels []string) error {
	if !hasClearImage {
		return errors.New("必须上传一张清晰图(type=clear)")
	}
	if hazyCount == 0 {
		return errors.New("至少上传一张有雾图(type=hazy)")
	}
	return v.ValidateHazyImageCount(hazyCount, hazeLevels)
}

// GetImageDimensions 获取图片尺寸的便捷方法
func (v *PairedImageValidator) GetImageDimensions(filePath string) (width, height int, err error) {
	info, err := v.GetImageInfo(filePath)
	if err != nil {
		return 0, 0, err
	}
	return info.Width, info.Height, nil
}

// ValidateImage 校验图片文件的基本信息
func (v *PairedImageValidator) ValidateImage(filePath string) error {
	// 校验文件是否存在
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return errors.New("文件不存在")
		}
		return fmt.Errorf("无法访问文件: %w", err)
	}

	// 校验文件大小（单个文件不超过 10MB，与文档规格一致）
	const maxSize = 10 * 1024 * 1024
	if fileInfo.Size() == 0 {
		return errors.New("文件为空")
	}
	if fileInfo.Size() > maxSize {
		return errors.New("文件太大，超过10MB限制")
	}

	// 校验图片类型
	if err := v.ValidateImageType(filePath); err != nil {
		return err
	}

	// 尝试解码图片以验证完整性
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("无法打开文件: %w", err)
	}
	defer file.Close()

	_, _, err = image.DecodeConfig(file)
	if err != nil {
		return fmt.Errorf("图片文件已损坏或格式不正确: %w", err)
	}

	return nil
}
