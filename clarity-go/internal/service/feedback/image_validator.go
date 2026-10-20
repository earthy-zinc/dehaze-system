package feedback

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/earthyzinc/clarity-go/pkg/common"
	"github.com/earthyzinc/clarity-go/pkg/config"
	"github.com/earthyzinc/clarity-go/pkg/storage"
)

var allowedImageExts = map[string]bool{
	".jpg":  true,
	".jpeg": true,
	".png":  true,
	".webp": true,
}

// validateImageUrls 校验图片 URL：数量、域名（各存储后端 baseUrl 前缀）、扩展名
// 与 Java/Python 对齐：域名按完整前缀比对（非仅 host），扩展名不合法报 A0701
func validateImageUrls(urls []string, maxCount int) error {
	if len(urls) > maxCount {
		return common.NewBizError(common.PARAM_ERROR, fmt.Sprintf("图片数量不能超过%d张", maxCount))
	}
	allowedPrefixes := collectAllowedStoragePrefixes()
	for _, u := range urls {
		if strings.TrimSpace(u) == "" {
			return common.NewBizError(common.PARAM_ERROR, "图片URL不能为空")
		}
		matched := false
		for _, prefix := range allowedPrefixes {
			if strings.HasPrefix(u, prefix) {
				matched = true
				break
			}
		}
		if !matched {
			return common.NewBizError(common.PARAM_ERROR, "图片URL域名不合法")
		}
		path := strings.ToLower(strings.SplitN(u, "#", 2)[0])
		path = strings.SplitN(path, "?", 2)[0]
		if !allowedImageExts[filepath.Ext(path)] {
			return common.NewBizError(
				common.USER_UPLOAD_FILE_TYPE_NOT_MATCH,
				common.USER_UPLOAD_FILE_TYPE_NOT_MATCH.Msg,
			)
		}
	}
	return nil
}

// collectAllowedStoragePrefixes 收集各存储后端对外访问地址前缀：
// MinIO 直连地址由 endpoint + bucket 拼出（无独立 baseUrl 配置），local/nginx-static 取配置的 baseUrl
func collectAllowedStoragePrefixes() []string {
	cfg := config.GetConfig()
	if cfg == nil {
		return nil
	}
	var prefixes []string
	for _, baseURL := range []string{
		storage.MinioBaseURL(cfg.File.Storage.MinIO.Endpoint, cfg.File.Storage.MinIO.BucketName),
		cfg.File.Storage.Local.BaseURL,
		cfg.File.Storage.NginxStatic.BaseURL,
	} {
		if baseURL = strings.TrimSpace(baseURL); baseURL == "" {
			continue
		}
		prefixes = append(prefixes, strings.TrimSuffix(baseURL, "/")+"/")
	}
	return prefixes
}
