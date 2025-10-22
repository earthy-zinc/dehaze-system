package service

import (
	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/dto"
	"github.com/earthyzinc/dehaze-go/model/vo"
	"github.com/earthyzinc/dehaze-go/utils"
	"gorm.io/gorm"
)

type ItemFileService struct{}

// SaveItemFile 保存项文件
func (itemFileService *ItemFileService) SaveItemFile(itemId int64, itemBO bo.DatasetItemBO) (imageFileInfo dto.ImageFileInfo, err error) {
	// 创建文件记录
	sysFileService := SysFileService{}

	// TODO: 这里需要根据实际的文件上传逻辑来实现
	// 暂时创建一个简单的文件对象
	fileBO := bo.FileBO{
		Name:       itemBO.Name,
		ObjectName: "",
		Extension:  itemBO.Extension,
		MD5:        "",
		Path:       "",
		Size:       itemBO.Size,
		URL:        "",
	}

	sysFile, err := sysFileService.SaveFile(fileBO)
	if err != nil {
		return imageFileInfo, err
	}

	// 创建项文件关联记录
	sysItemFile := model.SysItemFile{
		ItemID:      itemId,
		FileID:      int64(sysFile.ID),
		Type:        itemBO.Type,
		Description: utils.StringPtr(itemBO.Description),
	}

	err = global.DB.Create(&sysItemFile).Error
	if err != nil {
		return imageFileInfo, err
	}

	// 构建返回对象
	imageFileInfo = dto.ImageFileInfo{
		ID:            sysItemFile.ID,
		DatasetItemID: itemId,
		FileID:        int64(sysFile.ID),
		Type:          sysItemFile.Type,
		Description:   utils.StringVal(sysItemFile.Description),
		URL:           utils.StringVal(sysFile.URL),
	}

	return imageFileInfo, nil
}

// GetImageUrlVOs 获取图片URL VO列表
func (itemFileService *ItemFileService) GetImageUrlVOs(itemId int64) (imageUrlVOs []vo.ImageUrlVO, err error) {
	var sysItemFiles []model.SysItemFile
	err = global.DB.Where("item_id = ?", itemId).Find(&sysItemFiles).Error
	if err != nil {
		return imageUrlVOs, err
	}

	// 获取关联的文件信息
	for _, itemFile := range sysItemFiles {
		var sysFile model.SysFile
		err = global.DB.Where("id = ?", itemFile.FileID).First(&sysFile).Error
		if err != nil {
			continue
		}

		imageUrlVO := vo.ImageUrlVO{
			ID:          itemFile.ID,
			Type:        itemFile.Type,
			URL:         utils.StringVal(sysFile.URL),
			OriginURL:   utils.StringVal(sysFile.URL), // TODO: 实际应根据缩略图文件ID获取缩略图URL
			Description: utils.StringVal(itemFile.Description),
		}
		imageUrlVOs = append(imageUrlVOs, imageUrlVO)
	}

	return imageUrlVOs, nil
}

// DeleteItemFile 删除项文件
func (itemFileService *ItemFileService) DeleteItemFile(itemId int64) (err error) {
	result := global.DB.Delete(&model.SysItemFile{}, itemId)
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return gorm.ErrRecordNotFound
	}
	return nil
}

// DeleteItemFileByItemId 根据项ID删除项文件
func (itemFileService *ItemFileService) DeleteItemFileByItemId(itemId int64) (err error) {
	result := global.DB.Where("item_id = ?", itemId).Delete(&model.SysItemFile{})
	if result.Error != nil {
		return result.Error
	}
	return nil
}

// GetItemFileById 根据ID获取项文件
func (itemFileService *ItemFileService) GetItemFileById(itemFileId int64) (sysItemFile model.SysItemFile, err error) {
	err = global.DB.Where("id = ?", itemFileId).First(&sysItemFile).Error
	return sysItemFile, err
}
