package storage

import (
	"fmt"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
)

// NewStorage 根据配置创建存储服务实例
// cfgType: 存储类型（"minio" 或 "local"）
// minioCfg: MinIO 配置（仅当 cfgType 为 "minio" 时使用）
// localCfg: 本地存储配置（仅当 cfgType 为 "local" 时使用）
func NewStorage(cfgType string, minioCfg options.FileMinIO, localCfg options.FileLocal) (StorageService, error) {
	switch cfgType {
	case "minio":
		return NewMinioStorage(minioCfg)
	case "local":
		return NewLocalStorage(localCfg)
	default:
		return nil, fmt.Errorf("不支持的存储类型: %s，可选: minio, local", cfgType)
	}
}
