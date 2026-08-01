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

// validateImageUrls 校验图片 URL：数量、格式、扩展名、host 必须为已配置的存储后端 baseUrl host
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

// collectAllowedStorageHosts 收集所有已配置存储后端 baseUrl 的 host
func collectAllowedStorageHosts() map[string]bool {
	hosts := make(map[string]bool)
	cfg := config.GetConfig()
	if cfg == nil {
		return hosts
	}
	for _, baseURL := range []string{
		cfg.File.Storage.MinIO.BaseURL,
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
