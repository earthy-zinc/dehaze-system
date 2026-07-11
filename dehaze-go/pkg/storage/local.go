package storage

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
)

// LocalStorageService 本地文件系统存储实现
type LocalStorageService struct {
	rootPath string
}

// NewLocalStorage 创建本地存储服务实例
func NewLocalStorage(cfg options.FileLocal) (*LocalStorageService, error) {
	if cfg.UploadPath == "" {
		cfg.UploadPath = "./data/upload"
	}
	absPath, err := filepath.Abs(cfg.UploadPath)
	if err != nil {
		return nil, fmt.Errorf("解析本地存储路径失败: %w", err)
	}
	if err := os.MkdirAll(absPath, 0755); err != nil {
		return nil, fmt.Errorf("创建本地存储目录失败: %w", err)
	}
	return &LocalStorageService{rootPath: absPath}, nil
}

func (s *LocalStorageService) fullPath(objectName string) string {
	return filepath.Join(s.rootPath, objectName)
}

func (s *LocalStorageService) Upload(ctx context.Context, objectName string, reader io.Reader, size int64, contentType string) error {
	fullPath := s.fullPath(objectName)
	dir := filepath.Dir(fullPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("创建目录失败: %w", err)
	}

	file, err := os.Create(fullPath)
	if err != nil {
		return fmt.Errorf("创建文件失败: %w", err)
	}
	defer file.Close()

	if _, err := io.Copy(file, reader); err != nil {
		return fmt.Errorf("写入文件失败: %w", err)
	}
	return nil
}

func (s *LocalStorageService) Download(ctx context.Context, objectName string) (io.ReadCloser, error) {
	fullPath := s.fullPath(objectName)
	file, err := os.Open(fullPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("文件不存在: %s", objectName)
		}
		return nil, fmt.Errorf("打开文件失败: %w", err)
	}
	return file, nil
}

func (s *LocalStorageService) Delete(ctx context.Context, objectName string) error {
	fullPath := s.fullPath(objectName)
	if err := os.Remove(fullPath); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("删除文件失败: %w", err)
	}
	return nil
}

func (s *LocalStorageService) Exists(ctx context.Context, objectName string) (bool, error) {
	fullPath := s.fullPath(objectName)
	_, err := os.Stat(fullPath)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

func (s *LocalStorageService) GetURL(ctx context.Context, objectName string) (string, error) {
	return "", fmt.Errorf("本地存储不支持获取 URL，请通过 API 下载")
}
