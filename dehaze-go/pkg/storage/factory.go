package storage

import (
	"fmt"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
)

// StorageType 存储后端标识
const (
	StorageMinIO       = "minio"
	StorageLocal       = "local"
	StorageNginxStatic = "nginx-static"
)

// NewStorage 根据配置创建单个存储服务实例
// cfgType: 存储类型（minio/local/nginx-static）
func NewStorage(cfgType string, cfg options.FileStorageConfig) (StorageService, error) {
	switch cfgType {
	case StorageMinIO:
		return NewMinioStorage(cfg.MinIO)
	case StorageLocal:
		return NewLocalStorage(cfg.Local)
	case StorageNginxStatic:
		return NewNginxStorage(cfg.NginxStatic)
	default:
		return nil, fmt.Errorf("不支持的存储类型: %s，可选: minio, local, nginx-static", cfgType)
	}
}

// Registry 存储后端注册表，按 storage 标识取实例
// 文件管理模块只认 object_name + storage，通过 Registry 选后端，不感知具体实现。
type Registry struct {
	instances map[string]StorageService
	defaultType string
}

// NewRegistry 构建所有已配置的存储后端实例
// defaultType: 默认存储后端标识（上传时使用，当 sys_file.storage 为空时回退）
func NewRegistry(cfg options.File) (*Registry, error) {
	r := &Registry{
		instances: make(map[string]StorageService),
		defaultType: cfg.Type,
	}
	if cfg.Type == "" {
		r.defaultType = StorageMinIO
	}

	// MinIO 后端（需 endpoint 配置才创建）
	if cfg.Storage.MinIO.Endpoint != "" {
		svc, err := NewMinioStorage(cfg.Storage.MinIO)
		if err != nil {
			return nil, fmt.Errorf("初始化 MinIO 存储失败: %w", err)
		}
		r.instances[StorageMinIO] = svc
	}

	// 本地后端
	if cfg.Storage.Local.UploadPath != "" || cfg.Storage.Local.BaseURL != "" {
		svc, err := NewLocalStorage(cfg.Storage.Local)
		if err != nil {
			return nil, fmt.Errorf("初始化本地存储失败: %w", err)
		}
		r.instances[StorageLocal] = svc
	}

	// nginx 静态后端
	if cfg.Storage.NginxStatic.BaseURL != "" {
		svc, err := NewNginxStorage(cfg.Storage.NginxStatic)
		if err != nil {
			return nil, fmt.Errorf("初始化 nginx 静态存储失败: %w", err)
		}
		r.instances[StorageNginxStatic] = svc
	}

	if r.defaultType != "" {
		if _, ok := r.instances[r.defaultType]; !ok {
			return nil, fmt.Errorf("默认存储后端 %s 未配置或初始化失败", r.defaultType)
		}
	}
	if len(r.instances) == 0 {
		return nil, fmt.Errorf("未配置任何存储后端")
	}
	return r, nil
}

// Get 按 storage 标识取实例，storage 为空时取默认后端
func (r *Registry) Get(storage string) (StorageService, error) {
	if storage == "" {
		storage = r.defaultType
	}
	svc, ok := r.instances[storage]
	if !ok {
		return nil, fmt.Errorf("不支持的存储后端: %s", storage)
	}
	return svc, nil
}

// Default 取默认存储后端实例
func (r *Registry) Default() (StorageService, error) {
	return r.Get(r.defaultType)
}

// DefaultType 返回默认存储后端标识
func (r *Registry) DefaultType() string {
	return r.defaultType
}
