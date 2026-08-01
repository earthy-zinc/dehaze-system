package storage

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
)

// NginxStorageService nginx 静态服务后端实现
// 适用于 nginx 直服的静态文件（数据集、模型权重等），object_name 含资源前缀（如 datasets/...）
// Download = HTTP GET {baseUrl}/{objectName}；GetURL = {baseUrl}/{objectName}
type NginxStorageService struct {
	baseURL string
	client  *http.Client
}

// NewNginxStorage 创建 nginx 静态存储服务实例
func NewNginxStorage(cfg options.FileNginxStatic) (*NginxStorageService, error) {
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("nginx 静态存储 baseUrl 未配置")
	}
	return &NginxStorageService{
		baseURL: cfg.BaseURL,
		client:  &http.Client{Timeout: 30 * time.Second},
	}, nil
}

func (s *NginxStorageService) buildURL(objectName string) string {
	return strings.TrimRight(s.baseURL, "/") + "/" + strings.TrimPrefix(objectName, "/")
}

// Upload nginx 静态后端为只读直服，不支持上传
func (s *NginxStorageService) Upload(ctx context.Context, objectName string, reader io.Reader, size int64, contentType string) error {
	return fmt.Errorf("nginx 静态存储不支持上传")
}

// Download HTTP GET 取文件流
func (s *NginxStorageService) Download(ctx context.Context, objectName string) (io.ReadCloser, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.buildURL(objectName), nil)
	if err != nil {
		return nil, fmt.Errorf("构建 nginx 下载请求失败: %w", err)
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("nginx 下载失败: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("nginx 下载失败: HTTP %d", resp.StatusCode)
	}
	return resp.Body, nil
}

// Delete nginx 静态后端为只读，不支持删除
func (s *NginxStorageService) Delete(ctx context.Context, objectName string) error {
	return fmt.Errorf("nginx 静态存储不支持删除")
}

// Exists HTTP HEAD 校验文件是否存在
func (s *NginxStorageService) Exists(ctx context.Context, objectName string) (bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, s.buildURL(objectName), nil)
	if err != nil {
		return false, fmt.Errorf("构建 nginx HEAD 请求失败: %w", err)
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return false, fmt.Errorf("nginx 校验失败: %w", err)
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK, nil
}

// GetURL 运行时拼接：{baseUrl}/{objectName}
func (s *NginxStorageService) GetURL(ctx context.Context, objectName string) (string, error) {
	return s.buildURL(objectName), nil
}
