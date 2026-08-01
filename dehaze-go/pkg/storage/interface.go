package storage

import (
	"context"
	"io"
)

// StorageService 存储服务接口
// 定义统一的物理文件存储契约，支持 MinIO、本地文件系统、nginx 静态服务等不同后端。
// GetURL 为唯一 URL 生成出口：url = storage.baseUrl + "/" + object_name，运行时拼接，不落库。
type StorageService interface {
	// Upload 上传文件到存储后端（幂等：相同 objectName 覆盖写入）
	// reader: 文件内容流（调用方负责关闭）
	// objectName: 对象名（含路径，如 "upload/20250711/abc123.jpg"）
	// size: 文件大小（字节，-1 表示未知）
	// contentType: MIME 类型（如 "image/jpeg"）
	Upload(ctx context.Context, objectName string, reader io.Reader, size int64, contentType string) error

	// Download 获取文件内容流（调用方负责关闭）
	Download(ctx context.Context, objectName string) (io.ReadCloser, error)

	// Delete 删除物理文件（文件不存在时视为成功）
	Delete(ctx context.Context, objectName string) error

	// Exists 检查文件是否存在
	Exists(ctx context.Context, objectName string) (bool, error)

	// GetURL 运行时拼接文件访问地址（唯一 URL 生成出口）
	// url = storage.baseUrl.rstrip("/") + "/" + object_name
	GetURL(ctx context.Context, objectName string) (string, error)
}
