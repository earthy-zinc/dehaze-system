package file

import (
	"context"
	"crypto/md5"
	"errors"
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
	"github.com/dustin/go-humanize"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// FileService 文件服务
type FileService struct {
	fileRepo       filerepo.IFileRepository
	storageRegistry *storage.Registry
}

// NewFileService 创建文件服务实例
func NewFileService(fileRepo filerepo.IFileRepository, storageRegistry *storage.Registry) *FileService {
	return &FileService{
		fileRepo:       fileRepo,
		storageRegistry: storageRegistry,
	}
}

// UploadFile 上传文件（计算 MD5 → 去重 → 物理存储 → 写入元数据）
// reader: 已打开的文件流（由 API 层计算 MD5 后传入，reader 内部已定位到开头）
func (s *FileService) UploadFile(ctx context.Context, fileHeader *multipart.FileHeader, reader io.Reader, md5Hash string) (model.SysFile, error) {
	// 1. 构建 objectName
	now := time.Now()
	extension := filepath.Ext(fileHeader.Filename)
	uploadPath := fmt.Sprintf("upload/%s", now.Format("20060102"))
	objectName := fmt.Sprintf("%s/%s%s", uploadPath, md5Hash, extension)

	// 2. MD5 去重判断（仅查未删除记录）
	existingFile, err := s.fileRepo.FindByMD5(ctx, md5Hash)
	if err == nil && existingFile != nil {
		logger.Debug("文件秒传命中", zap.String("md5", md5Hash), zap.Int64("fileID", existingFile.ID))
		return *existingFile, nil
	}

	// 3. 物理存储（用默认后端上传）
	storageType := s.storageRegistry.DefaultType()
	storageSvc, err := s.storageRegistry.Get(storageType)
	if err != nil {
		return model.SysFile{}, common.WrapBizError(common.FILE_UPLOAD_FAILED, "存储后端不可用", err)
	}
	contentType := fileHeader.Header.Get("Content-Type")
	if err := storageSvc.Upload(ctx, objectName, reader, fileHeader.Size, contentType); err != nil {
		logger.Error("物理文件存储失败", zap.String("objectName", objectName), zap.Error(err))
		return model.SysFile{}, common.WrapBizError(common.FILE_UPLOAD_FAILED, "文件存储失败", err)
	}

	// 4. 写入元数据
	fileSize := fileHeader.Size
	sysFile := model.SysFile{
		BaseModel:  model.BaseModel{CreatedAt: now, UpdatedAt: now},
		Type:       utils.StringPtr(extension),
		Name:       fileHeader.Filename,
		ObjectName: objectName,
		Storage:    storageType,
		Size:       humanize.Bytes(uint64(fileSize)),
		SizeBytes:  &fileSize,
		MD5:        md5Hash,
	}

	if err := s.fileRepo.Upsert(ctx, &sysFile); err != nil {
		logger.Error("文件元数据写入失败（孤儿文件待清理）", zap.String("objectName", objectName), zap.Error(err))
		return model.SysFile{}, common.WrapBizError(common.DATABASE_ERROR, "保存文件记录失败", err)
	}

	logger.Debug("文件上传成功", zap.Int64("fileID", sysFile.ID), zap.String("md5", md5Hash))
	return sysFile, nil
}

// CheckFile 校验文件是否存在
func (s *FileService) CheckFile(ctx context.Context, md5 string) (*model.SysFile, error) {
	existingFile, err := s.fileRepo.FindByMD5(ctx, md5)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询文件失败", err)
	}
	return existingFile, nil
}

// DeleteFile 删除文件
func (s *FileService) DeleteFile(ctx context.Context, fileId int64) (err error) {
	file, err := s.fileRepo.FindByID(ctx, fileId)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "文件不存在")
		}
		return common.WrapBizError(common.DATABASE_ERROR, "查询文件失败", err)
	}

	// 删除元数据
	if err := s.fileRepo.Delete(ctx, []int64{fileId}); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除文件失败", err)
	}

	// 删除物理文件（失败不影响业务，孤儿文件由定时任务清理）
	if storageSvc, err := s.storageRegistry.Get(file.Storage); err == nil {
		if err := storageSvc.Delete(ctx, file.ObjectName); err != nil {
			logger.Warn("物理文件删除失败（孤儿文件待清理）", zap.String("objectName", file.ObjectName), zap.Error(err))
		}
	} else {
		logger.Warn("物理文件删除跳过：存储后端不可用", zap.String("storage", file.Storage), zap.Error(err))
	}

	logger.Debug("文件删除成功", zap.Int64("fileID", fileId))
	return nil
}

// GetFileById 根据ID获取文件
func (s *FileService) GetFileById(ctx context.Context, fileId int64) (sysFile model.SysFile, err error) {
	file, err := s.fileRepo.FindByID(ctx, fileId)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			// 文件不存在时返回空记录，由调用方判断，与 Java/Python 行为一致
			return model.SysFile{}, nil
		}
		return model.SysFile{}, common.WrapBizError(common.DATABASE_ERROR, "查询文件失败", err)
	}
	return *file, nil
}

// GetFilesByIdsMap 批量查询文件，返回 fileID → SysFile 的映射（用于消除 N+1 查询）
func (s *FileService) GetFilesByIdsMap(ctx context.Context, fileIDs []int64) (map[int64]model.SysFile, error) {
	if len(fileIDs) == 0 {
		return map[int64]model.SysFile{}, nil
	}
	files, err := s.fileRepo.FindByIDs(ctx, fileIDs)
	if err != nil {
		return nil, err
	}
	result := make(map[int64]model.SysFile, len(files))
	for i := range files {
		result[int64(files[i].ID)] = files[i]
	}
	return result, nil
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

// GetFileByObjectName 根据对象名获取文件记录
func (s *FileService) GetFileByObjectName(ctx context.Context, objectName string) (*model.SysFile, error) {
	return s.fileRepo.FindByObjectName(ctx, objectName)
}

// DownloadFile 下载文件（返回文件流）
func (s *FileService) DownloadFile(ctx context.Context, objectName string) (io.ReadCloser, *model.SysFile, error) {
	file, err := s.fileRepo.FindByObjectName(ctx, objectName)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "文件不存在")
		}
		return nil, nil, common.WrapBizError(common.DATABASE_ERROR, "查询文件失败", err)
	}

	storageSvc, err := s.storageRegistry.Get(file.Storage)
	if err != nil {
		return nil, nil, common.NewBizError(common.OBJECT_STORAGE_ERROR, "存储后端不可用")
	}
	reader, err := storageSvc.Download(ctx, objectName)
	if err != nil {
		return nil, nil, common.NewBizError(common.OBJECT_STORAGE_ERROR, "文件下载失败")
	}

	return reader, file, nil
}

// GetURL 运行时拼接文件访问 URL（唯一 URL 生成出口）
// 按 file.Storage 选后端，调用 storageService.GetURL(objectName)
func (s *FileService) GetURL(ctx context.Context, file *model.SysFile) string {
	if file == nil {
		return ""
	}
	storageSvc, err := s.storageRegistry.Get(file.Storage)
	if err != nil {
		return ""
	}
	url, err := storageSvc.GetURL(ctx, file.ObjectName)
	if err != nil {
		logger.Warn("拼接文件 URL 失败", zap.String("objectName", file.ObjectName), zap.Error(err))
		return ""
	}
	return url
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
