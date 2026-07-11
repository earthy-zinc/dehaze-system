package file

import (
	"context"
	"crypto/md5"
	"fmt"
	"io"
	"mime/multipart"
	"path/filepath"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	filerepo "github.com/earthyzinc/dehaze-go/internal/repository/file"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/storage"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// FileService 文件服务
type FileService struct {
	fileRepo       filerepo.IFileRepository
	storageService storage.StorageService
}

// NewFileService 创建文件服务实例
func NewFileService(fileRepo filerepo.IFileRepository, storageService storage.StorageService) *FileService {
	return &FileService{
		fileRepo:       fileRepo,
		storageService: storageService,
	}
}

// UploadFile 上传文件（计算 MD5 → 去重 → 物理存储 → 写入元数据）
// reader: 已打开的文件流（由 API 层计算 MD5 后传入，reader 内部已定位到开头）
func (s *FileService) UploadFile(ctx context.Context, fileHeader *multipart.FileHeader, reader io.Reader, md5Hash string, baseURL string) (model.SysFile, error) {
	// 1. 构建 FileBO
	now := time.Now()
	extension := filepath.Ext(fileHeader.Filename)
	uploadPath := fmt.Sprintf("upload/%s", now.Format("20060102"))
	objectName := fmt.Sprintf("%s/%s%s", uploadPath, md5Hash, extension)

	// 2. MD5 去重判断
	existingFile, err := s.fileRepo.FindByMD5(ctx, md5Hash)
	if err == nil && existingFile != nil {
		logger.Info("文件秒传命中", zap.String("md5", md5Hash), zap.Int("fileID", existingFile.ID))
		return *existingFile, nil
	}

	// 3. 物理存储
	contentType := fileHeader.Header.Get("Content-Type")
	if err := s.storageService.Upload(ctx, objectName, reader, fileHeader.Size, contentType); err != nil {
		logger.Error("物理文件存储失败", zap.String("objectName", objectName), zap.Error(err))
		return model.SysFile{}, common.WrapBizError(common.FILE_UPLOAD_FAILED, "文件存储失败", err)
	}

	// 4. 获取文件访问 URL
	fileURL, err := s.storageService.GetURL(ctx, objectName)
	if err != nil {
		logger.Warn("获取文件URL失败，使用 baseURL 拼接", zap.Error(err))
		fileURL = baseURL + "/" + objectName
	}

	// 5. 写入元数据（事务）
	sysFile := model.SysFile{
		Type:       utils.StringPtr(extension),
		URL:        utils.StringPtr(fileURL),
		Name:       fileHeader.Filename,
		ObjectName: objectName,
		Size:       fmt.Sprintf("%d", fileHeader.Size),
		Path:       uploadPath,
		MD5:        md5Hash,
		CreatedAt:  now,
		UpdatedAt:  now,
	}

	createdFile, err := s.fileRepo.Create(ctx, &sysFile)
	if err != nil {
		// 物理文件已存储，DB 写入失败产生孤儿文件，记录日志由定时任务清理
		logger.Error("文件元数据写入失败（孤儿文件待清理）", zap.String("objectName", objectName), zap.Error(err))
		return model.SysFile{}, common.WrapBizError(common.DATABASE_ERROR, "保存文件记录失败", err)
	}

	logger.Info("文件上传成功", zap.Int("fileID", createdFile.ID), zap.String("md5", md5Hash))
	return *createdFile, nil
}

// CheckFile 校验文件是否存在
func (s *FileService) CheckFile(md5 string) bool {
	ctx := context.Background()
	existingFile, err := s.fileRepo.FindByMD5(ctx, md5)
	return err == nil && existingFile != nil
}

// DeleteFile 删除文件
func (s *FileService) DeleteFile(fileId int64) (err error) {
	ctx := context.Background()

	file, err := s.fileRepo.FindByID(ctx, fileId)
	if err != nil {
		if err == gorm.ErrRecordNotFound {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "文件不存在")
		}
		return common.WrapBizError(common.DATABASE_ERROR, "查询文件失败", err)
	}

	// 删除元数据
	if err := s.fileRepo.Delete(ctx, []int64{fileId}); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除文件失败", err)
	}

	// 删除物理文件（失败不影响业务，孤儿文件由定时任务清理）
	if err := s.storageService.Delete(ctx, file.ObjectName); err != nil {
		logger.Warn("物理文件删除失败（孤儿文件待清理）", zap.String("objectName", file.ObjectName), zap.Error(err))
	}

	logger.Info("文件删除成功", zap.Int64("fileID", fileId))
	return nil
}

// GetFileById 根据ID获取文件
func (s *FileService) GetFileById(fileId int64) (sysFile model.SysFile, err error) {
	ctx := context.Background()

	file, err := s.fileRepo.FindByID(ctx, fileId)
	if err != nil {
		if err == gorm.ErrRecordNotFound {
			return model.SysFile{}, common.NewBizError(common.RESOURCE_NOT_FOUND, "文件不存在")
		}
		return model.SysFile{}, common.WrapBizError(common.DATABASE_ERROR, "查询文件失败", err)
	}
	return *file, nil
}

// GetPage 分页查询文件列表
func (s *FileService) GetPage(ctx context.Context, pageNum, pageSize int, keywords string) (*common.PageResult, error) {
	files, total, err := s.fileRepo.FindPage(ctx, pageNum, pageSize, keywords)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询文件列表失败", err)
	}
	return &common.PageResult{
		List:     files,
		Total:    total,
		Page:     pageNum,
		PageSize: pageSize,
	}, nil
}

// DownloadFile 下载文件（返回文件流）
func (s *FileService) DownloadFile(ctx context.Context, objectName string) (io.ReadCloser, *model.SysFile, error) {
	file, err := s.fileRepo.FindByObjectName(ctx, objectName)
	if err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "文件不存在")
		}
		return nil, nil, common.WrapBizError(common.DATABASE_ERROR, "查询文件失败", err)
	}

	reader, err := s.storageService.Download(ctx, objectName)
	if err != nil {
		return nil, nil, common.NewBizError(common.OBJECT_STORAGE_ERROR, "文件下载失败")
	}

	return reader, file, nil
}

// ComputeMD5 流式计算 MD5，返回 MD5 十六进制字符串和重置后的 reader
// 调用方传入已打开的 multipart.File，此函数会计算 MD5 后 Seek 回开头
// 返回的 reader 可直接用于上传存储
func ComputeMD5(reader io.ReadSeeker) (string, io.ReadSeeker, error) {
	hash := md5.New()
	if _, err := io.Copy(hash, reader); err != nil {
		return "", nil, fmt.Errorf("计算文件MD5失败: %w", err)
	}
	md5Hex := fmt.Sprintf("%x", hash.Sum(nil))

	// Seek 回文件开头，以便后续读取
	if _, err := reader.Seek(0, io.SeekStart); err != nil {
		return "", nil, fmt.Errorf("重置文件读取位置失败: %w", err)
	}

	return md5Hex, reader, nil
}
