package feedback

import (
	"net/url"
	"path/filepath"
	"strings"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/storage"
)

var allowedImageExts = map[string]bool{
	".jpg":  true,
	".jpeg": true,
	".png":  true,
	".webp": true,
}

// validateImageUrls 校验图片 URL：数量、格式、扩展名、host 必须为已配置存储后端的对外访问 host
func validateImageUrls(urls []string, maxCount int) error {
	if len(urls) > maxCount {
		return common.NewBizError(common.PARAM_ERROR, "图片校验失败：数量超过上限")
	}
	allowedHosts := collectAllowedStorageHosts()
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
		if len(allowedHosts) > 0 && !allowedHosts[parsed.Host] {
			return common.NewBizError(common.PARAM_ERROR, "图片校验失败：URL 必须为已配置的存储后端域名")
		}
	}
	return nil
}

// collectAllowedStorageHosts 收集各存储后端对外访问地址的 host：
// MinIO 直连地址由 endpoint + bucket 拼出（无独立 baseUrl 配置），local/nginx-static 取配置的 baseUrl
func collectAllowedStorageHosts() map[string]bool {
	hosts := make(map[string]bool)
	cfg := config.GetConfig()
	if cfg == nil {
		return hosts
	}
	for _, baseURL := range []string{
		storage.MinioBaseURL(cfg.File.Storage.MinIO.Endpoint, cfg.File.Storage.MinIO.BucketName),
		cfg.File.Storage.Local.BaseURL,
		cfg.File.Storage.NginxStatic.BaseURL,
	} {
		if baseURL == "" {
			continue
		}
		if parsed, err := url.Parse(baseURL); err == nil && parsed.Host != "" {
			hosts[parsed.Host] = true
		}
	}
	return hosts
}
