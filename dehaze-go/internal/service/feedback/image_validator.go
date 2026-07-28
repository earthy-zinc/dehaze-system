package feedback

import (
	"net/url"
	"path/filepath"
	"strings"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
)

var allowedImageExts = map[string]bool{
	".jpg":  true,
	".jpeg": true,
	".png":  true,
	".webp": true,
}

func validateImageUrls(urls []string, maxCount int) error {
	if len(urls) > maxCount {
		return common.NewBizError(common.PARAM_ERROR, "图片校验失败：数量超过上限")
	}
	minioHost := extractMinIOHost()
	for _, u := range urls {
		if u == "" {
			return common.NewBizError(common.PARAM_ERROR, "图片校验失败：URL 不能为空")
		}
		parsed, err := url.Parse(u)
		if err != nil || parsed.Host == "" {
			return common.NewBizError(common.PARAM_ERROR, "图片校验失败：URL 格式不正确")
		}
		ext := strings.ToLower(filepath.Ext(parsed.Path))
		if !allowedImageExts[ext] {
			return common.NewBizError(common.PARAM_ERROR, "图片校验失败：仅支持 jpg/jpeg/png/webp")
		}
		if minioHost != "" && parsed.Host != minioHost {
			return common.NewBizError(common.PARAM_ERROR, "图片校验失败：URL 必须为 MinIO 域名")
		}
	}
	return nil
}

func extractMinIOHost() string {
	cfg := config.GetConfig()
	if cfg == nil || cfg.File.Type != "minio" || cfg.File.MinIO.Endpoint == "" {
		return ""
	}
	endpoint := cfg.File.MinIO.Endpoint
	if i := strings.Index(endpoint, "://"); i >= 0 {
		endpoint = endpoint[i+3:]
	}
	if i := strings.Index(endpoint, "/"); i >= 0 {
		endpoint = endpoint[:i]
	}
	return endpoint
}
