package storage

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

// MinioStorageService MinIO 对象存储实现
type MinioStorageService struct {
	client     *minio.Client
	bucketName string
}

// NewMinioStorage 创建 MinIO 存储服务实例
func NewMinioStorage(cfg options.FileMinIO) (*MinioStorageService, error) {
	client, err := minio.New(cfg.Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(cfg.AccessKey, cfg.SecretKey, ""),
		Secure: strings.HasPrefix(cfg.Endpoint, "https://"),
	})
	if err != nil {
		return nil, fmt.Errorf("创建 MinIO 客户端失败: %w", err)
	}

	svc := &MinioStorageService{
		client:     client,
		bucketName: cfg.BucketName,
	}

	// 确保存储桶存在
	if err := svc.ensureBucket(context.Background()); err != nil {
		return nil, err
	}

	return svc, nil
}

// ensureBucket 确保存储桶存在，不存在则创建并设置公开读策略
func (s *MinioStorageService) ensureBucket(ctx context.Context) error {
	exists, err := s.client.BucketExists(ctx, s.bucketName)
	if err != nil {
		return fmt.Errorf("检查存储桶是否存在失败: %w", err)
	}
	if !exists {
		if err := s.client.MakeBucket(ctx, s.bucketName, minio.MakeBucketOptions{}); err != nil {
			return fmt.Errorf("创建存储桶失败: %w", err)
		}
		// 设置公开读策略
		policy := fmt.Sprintf(`{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":["*"]},"Action":["s3:GetObject"],"Resource":["arn:aws:s3:::%s/*"]}]}`, s.bucketName)
		if err := s.client.SetBucketPolicy(ctx, s.bucketName, policy); err != nil {
			return fmt.Errorf("设置存储桶策略失败: %w", err)
		}
	}
	return nil
}

// Upload 上传文件到 MinIO
func (s *MinioStorageService) Upload(ctx context.Context, objectName string, reader io.Reader, size int64, contentType string) error {
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	_, err := s.client.PutObject(ctx, s.bucketName, objectName, reader, size, minio.PutObjectOptions{
		ContentType: contentType,
	})
	if err != nil {
		return fmt.Errorf("MinIO 上传失败: %w", err)
	}
	return nil
}

// Download 从 MinIO 下载文件流
func (s *MinioStorageService) Download(ctx context.Context, objectName string) (io.ReadCloser, error) {
	obj, err := s.client.GetObject(ctx, s.bucketName, objectName, minio.GetObjectOptions{})
	if err != nil {
		return nil, fmt.Errorf("MinIO 下载失败: %w", err)
	}
	return obj, nil
}

// Delete 从 MinIO 删除文件
func (s *MinioStorageService) Delete(ctx context.Context, objectName string) error {
	err := s.client.RemoveObject(ctx, s.bucketName, objectName, minio.RemoveObjectOptions{})
	if err != nil {
		// 忽略文件不存在的错误
		if minio.ToErrorResponse(err).Code == "NoSuchKey" {
			return nil
		}
		return fmt.Errorf("MinIO 删除失败: %w", err)
	}
	return nil
}

// Exists 检查 MinIO 中文件是否存在
func (s *MinioStorageService) Exists(ctx context.Context, objectName string) (bool, error) {
	_, err := s.client.StatObject(ctx, s.bucketName, objectName, minio.StatObjectOptions{})
	if err != nil {
		if minio.ToErrorResponse(err).Code == "NoSuchKey" {
			return false, nil
		}
		return false, fmt.Errorf("MinIO 检查文件失败: %w", err)
	}
	return true, nil
}

// GetURL 获取 MinIO 文件访问 URL
func (s *MinioStorageService) GetURL(ctx context.Context, objectName string) (string, error) {
	// 返回对象路径（配合公开桶策略直接用 endpoint/bucket/object 访问）
	return fmt.Sprintf("%s/%s/%s", strings.TrimRight(s.client.EndpointURL().String(), "/"), s.bucketName, objectName), nil
}
