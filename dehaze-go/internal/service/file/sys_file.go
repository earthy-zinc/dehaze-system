package file

import (
	"context"
	"fmt"
	"mime/multipart"
	"path/filepath"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	filerepo "github.com/earthyzinc/dehaze-go/internal/repository/file"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"gorm.io/gorm"
)

// FileService 文件服务
type FileService struct {
	fileRepo filerepo.IFileRepository
}

// NewFileService 创建文件服务实例
func NewFileService(fileRepo filerepo.IFileRepository) *FileService {
	return &FileService{fileRepo: fileRepo}
}

// UploadFile 上传文件（构建 FileBO）
func (s *FileService) UploadFile(file *multipart.FileHeader, baseUrl, uploadPath string) (fileBO bo.FileBO, err error) {
	// TODO: 实现文件上传逻辑
	// 这里需要根据实际的文件存储服务（如本地存储、OSS等）来实现
	// 目前仅构建FileBO对象
	fileBO.Name = file.Filename
	fileBO.Extension = filepath.Ext(file.Filename)
	fileBO.Size = file.Size
	fileBO.Path = uploadPath
	fileBO.URL = baseUrl + "/" + uploadPath + "/" + file.Filename
	fileBO.ObjectName = uploadPath + "/" + file.Filename
	// MD5计算需要实际读取文件内容，这里暂时留空
	// fileBO.MD5 =

	return fileBO, nil
}

// SaveFile 保存文件信息到数据库
func (s *FileService) SaveFile(fileBO bo.FileBO) (sysFile model.SysFile, err error) {
	// 先根据md5查询，如果存在则直接返回
	ctx := context.Background()
	existingFile, err := s.fileRepo.FindByMD5(ctx, fileBO.MD5)
	if err == nil && existingFile != nil {
		return *existingFile, nil
	}

	// 如果不存在，则保存文件信息到数据库
	sysFile = model.SysFile{
		Type:       utils.StringPtr(fileBO.Extension),
		URL:        utils.StringPtr(fileBO.URL),
		Name:       fileBO.Name,
		ObjectName: fileBO.ObjectName,
		Size:       fmt.Sprintf("%d", fileBO.Size),
		Path:       fileBO.Path,
		MD5:        fileBO.MD5,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}

	createdFile, err := s.fileRepo.Create(ctx, &sysFile)
	if err != nil {
		return model.SysFile{}, common.WrapBizError(common.DATABASE_ERROR, "保存文件记录失败", err)
	}
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

	// 先检查文件是否存在
	_, err = s.fileRepo.FindByID(ctx, fileId)
	if err != nil {
		if err == gorm.ErrRecordNotFound {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "文件不存在")
		}
		return common.WrapBizError(common.DATABASE_ERROR, "查询文件失败", err)
	}

	if err := s.fileRepo.Delete(ctx, []int64{fileId}); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除文件失败", err)
	}
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

// DownloadFile 下载文件
func (s *FileService) DownloadFile(objectName string) (filePath string, err error) {
	// TODO: 实现文件下载逻辑
	// 这里需要根据实际的文件存储服务（如本地存储、OSS等）来实现
	// 目前仅返回文件路径
	ctx := context.Background()

	file, err := s.fileRepo.FindByObjectName(ctx, objectName)
	if err != nil {
		if err == gorm.ErrRecordNotFound {
			return "", common.NewBizError(common.RESOURCE_NOT_FOUND, "文件不存在")
		}
		return "", common.WrapBizError(common.DATABASE_ERROR, "查询文件失败", err)
	}

	return file.Path, nil
}
