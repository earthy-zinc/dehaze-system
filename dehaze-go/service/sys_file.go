package service

import (
	"fmt"
	"mime/multipart"
	"path/filepath"
	"time"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"gorm.io/gorm"
)

type FileService struct{}

type SysFileService struct{}

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
	// 先根据md5查询，如果存在则直接返回
	var existingFile model.SysFile
	result := global.DB.Where("md5 = ?", fileBO.MD5).First(&existingFile)
	if result.Error == nil {
		return existingFile, nil
	}

	// 如果不存在，则保存文件信息到数据库
	sysFile = model.SysFile{
		Type:       fileBO.Extension,
		URL:        fileBO.URL,
		Name:       fileBO.Name,
		ObjectName: fileBO.ObjectName,
		Size:       fmt.Sprintf("%d", fileBO.Size),
		Path:       fileBO.Path,
		MD5:        fileBO.MD5,
		BaseModel: model.BaseModel{
			ID:        0,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
	}

	err = global.DB.Create(&sysFile).Error
	return sysFile, err
}

// CheckFile 校验文件是否存在
func (sysFileService *SysFileService) CheckFile(md5 string) bool {
	var count int64
	global.DB.Model(&model.SysFile{}).Where("md5 = ?", md5).Count(&count)
	return count > 0
}

// DeleteFile 删除文件
func (sysFileService *SysFileService) DeleteFile(fileId int64) (err error) {
	result := global.DB.Where("id = ?", fileId).Delete(&model.SysFile{})
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return gorm.ErrRecordNotFound
	}
	return nil
}

// GetFileById 根据ID获取文件
func (sysFileService *SysFileService) GetFileById(fileId int64) (sysFile model.SysFile, err error) {
	err = global.DB.Where("id = ?", fileId).First(&sysFile).Error
	return sysFile, err
}

// DownloadFile 下载文件
func (sysFileService *SysFileService) DownloadFile(objectName string) (filePath string, err error) {
	// TODO: 实现文件下载逻辑
	// 这里需要根据实际的文件存储服务（如本地存储、OSS等）来实现
	// 目前仅返回文件路径
	var sysFile model.SysFile
	err = global.DB.Where("object_name = ?", objectName).First(&sysFile).Error
	if err != nil {
		return "", err
	}
	
	return sysFile.Path, nil
}