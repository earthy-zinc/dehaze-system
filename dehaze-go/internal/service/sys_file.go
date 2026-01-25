package service

import (
	"context"
	"fmt"
	"mime/multipart"
	"path/filepath"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/global"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"gorm.io/gorm"
)

type FileService struct{}

type SysFileService struct {
	fileRepo repository.IFileRepository
}

// NewSysFileService 创建文件服务实例
func NewSysFileService(fileRepo repository.IFileRepository) *SysFileService {
	return &SysFileService{fileRepo: fileRepo}
}

// getRepo 获取 Repository（兼容零值实例）
func (s *SysFileService) getRepo() repository.IFileRepository {
	if s.fileRepo != nil {
		return s.fileRepo
	}
	return repository.NewFileRepository(global.DB)
}

// SetFileRepo 设置 Repository（测试用）
func (s *SysFileService) SetFileRepo(repo repository.IFileRepository) {
	s.fileRepo = repo
}

// UploadFile 上传文件
func (fileService *FileService) UploadFile(file *multipart.FileHeader, baseUrl, uploadPath string) (fileBO bo.FileBO, err error) {
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
func (sysFileService *SysFileService) SaveFile(fileBO bo.FileBO) (sysFile model.SysFile, err error) {
	repo := sysFileService.getRepo()

	// 先根据md5查询，如果存在则直接返回
	ctx := context.Background()
	existingFile, err := repo.FindByMD5(ctx, fileBO.MD5)
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

	createdFile, err := repo.Create(ctx, &sysFile)
	if err != nil {
		return model.SysFile{}, err
	}
	return *createdFile, nil
}

// CheckFile 校验文件是否存在
func (sysFileService *SysFileService) CheckFile(md5 string) bool {
	ctx := context.Background()
	existingFile, err := sysFileService.getRepo().FindByMD5(ctx, md5)
	return err == nil && existingFile != nil
}

// DeleteFile 删除文件
func (sysFileService *SysFileService) DeleteFile(fileId int64) (err error) {
	ctx := context.Background()
	repo := sysFileService.getRepo()

	// 先检查文件是否存在
	_, err = repo.FindByID(ctx, fileId)
	if err != nil {
		if err == gorm.ErrRecordNotFound {
			return gorm.ErrRecordNotFound
		}
		return err
	}

	return repo.Delete(ctx, []int64{fileId})
}

// GetFileById 根据ID获取文件
func (sysFileService *SysFileService) GetFileById(fileId int64) (sysFile model.SysFile, err error) {
	ctx := context.Background()
	repo := sysFileService.getRepo()

	file, err := repo.FindByID(ctx, fileId)
	if err != nil {
		return model.SysFile{}, err
	}
	return *file, nil
}

// DownloadFile 下载文件
func (sysFileService *SysFileService) DownloadFile(objectName string) (filePath string, err error) {
	// TODO: 实现文件下载逻辑
	// 这里需要根据实际的文件存储服务（如本地存储、OSS等）来实现
	// 目前仅返回文件路径
	ctx := context.Background()
	repo := sysFileService.getRepo()

	file, err := repo.FindByObjectName(ctx, objectName)
	if err != nil {
		return "", err
	}

	return file.Path, nil
}
