package service

import (
	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"gorm.io/gorm"
)

type DatasetItemService struct{}

// CreateDatasetItem 创建数据集项
func (datasetItemService *DatasetItemService) CreateDatasetItem(datasetId int64) (sysDatasetItem model.SysDatasetItem, err error) {
	sysDatasetItem = model.SysDatasetItem{
		DatasetID: datasetId,
		Name:      "",
	}
	err = global.DB.Create(&sysDatasetItem).Error
	return sysDatasetItem, err
}

// CreateDatasetItemWithName 创建带名称的数据集项
func (datasetItemService *DatasetItemService) CreateDatasetItemWithName(datasetId int64, itemName string) (sysDatasetItem model.SysDatasetItem, err error) {
	sysDatasetItem = model.SysDatasetItem{
		DatasetID: datasetId,
		Name:      itemName,
	}
	err = global.DB.Create(&sysDatasetItem).Error
	return sysDatasetItem, err
}

// DeleteDatasetItem 删除数据集项
func (datasetItemService *DatasetItemService) DeleteDatasetItem(datasetItemId int64) (err error) {
	// 先删除关联的项文件
	itemFileService := ItemFileService{}
	err = itemFileService.DeleteItemFileByItemId(datasetItemId)
	if err != nil {
		return err
	}

	// 删除数据集项本身
	result := global.DB.Delete(&model.SysDatasetItem{}, datasetItemId)
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return gorm.ErrRecordNotFound
	}
	return nil
}

// UpdateDatasetItem 更新数据集项
func (datasetItemService *DatasetItemService) UpdateDatasetItem(datasetItemId int64, itemName string) (err error) {
	updates := map[string]interface{}{
		"name": itemName,
	}
	result := global.DB.Model(&model.SysDatasetItem{}).Where("id = ?", datasetItemId).Updates(updates)
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return gorm.ErrRecordNotFound
	}
	return nil
}

// GetDatasetItemById 根据ID获取数据集项
func (datasetItemService *DatasetItemService) GetDatasetItemById(datasetItemId int64) (sysDatasetItem model.SysDatasetItem, err error) {
	err = global.DB.Where("id = ?", datasetItemId).First(&sysDatasetItem).Error
	return sysDatasetItem, err
}
