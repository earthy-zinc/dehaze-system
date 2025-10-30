package service

import (
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/model/vo"
	"gorm.io/gorm"
)

type DatasetService struct{}

// GetDatasetList 获取数据集列表
func (datasetService *DatasetService) GetDatasetList(queryParams query.DatasetQuery) (datasetVOs []vo.DatasetVO, err error) {
	// 构建查询
	db := global.DB.Model(&model.SysDataset{}).
		Where("deleted = ?", 0)

	// 添加查询条件
	if queryParams.Keywords != "" {
		keyword := "%" + queryParams.Keywords + "%"
		db = db.Where("name LIKE ?", keyword)
	}

	// 查询数据
	var datasetList []model.SysDataset
	err = db.Find(&datasetList).Error
	if err != nil {
		return datasetVOs, err
	}

	if len(datasetList) == 0 {
		return datasetVOs, nil
	}

	// 获取所有数据集ID
	datasetIds := make(map[int64]bool)
	for _, dataset := range datasetList {
		datasetIds[dataset.ID] = true
	}

	// 获取父节点ID
	parentIds := make(map[int64]bool)
	for _, dataset := range datasetList {
		parentIds[dataset.ParentID] = true
	}

	// 获取根节点ID（递归的起点），即父节点ID中不包含在数据集ID中的节点
	var rootIds []int64
	for parentId := range parentIds {
		if _, exists := datasetIds[parentId]; !exists {
			rootIds = append(rootIds, parentId)
		}
	}

	// 递归生成数据集树形列表
	for _, rootId := range rootIds {
		children := datasetService.recurDatasetList(rootId, datasetList)
		datasetVOs = append(datasetVOs, children...)
	}

	return datasetVOs, nil
}

// 递归生成数据集树形列表
func (datasetService *DatasetService) recurDatasetList(parentId int64, datasetList []model.SysDataset) []vo.DatasetVO {
	var result []vo.DatasetVO
	for _, dataset := range datasetList {
		if dataset.ParentID == parentId {
			datasetVO := vo.DatasetVO{
				ID:          dataset.ID,
				ParentID:    dataset.ParentID,
				Type:        dataset.Type,
				Name:        dataset.Name,
				Description: dataset.Description,
				Path:        dataset.Path,
				Size:        dataset.Size,
				CreateTime:  dataset.CreatedAt,
				UpdateTime:  dataset.UpdatedAt,
				Status:      int(dataset.Status),
			}
			// 递归获取子数据集
			children := datasetService.recurDatasetList(dataset.ID, datasetList)
			datasetVO.Children = children
			result = append(result, datasetVO)
		}
	}
	return result
}

// GetDatasetOptions 数据集下拉选项
func (datasetService *DatasetService) GetDatasetOptions() (options []vo.Option, err error) {
	// 查询启用状态的数据集数据
	var datasetList []model.SysDataset
	err = global.DB.Model(&model.SysDataset{}).
		Where("status = ? AND deleted = ?", 1, 0).
		Select("id, parent_id, name").
		Find(&datasetList).Error

	if err != nil {
		return options, err
	}

	if len(datasetList) == 0 {
		return options, nil
	}

	// 获取所有数据集ID
	datasetIds := make(map[int64]bool)
	for _, dataset := range datasetList {
		datasetIds[dataset.ID] = true
	}

	// 获取父节点ID
	parentIds := make(map[int64]bool)
	for _, dataset := range datasetList {
		parentIds[dataset.ParentID] = true
	}

	// 获取根节点ID
	var rootIds []int64
	for parentId := range parentIds {
		if _, exists := datasetIds[parentId]; !exists {
			rootIds = append(rootIds, parentId)
		}
	}

	// 递归生成数据集树形下拉选项
	for _, rootId := range rootIds {
		children := datasetService.recurDatasetTreeOptions(rootId, datasetList)
		options = append(options, children...)
	}

	return options, nil
}

// 递归生成数据集树形下拉选项
func (datasetService *DatasetService) recurDatasetTreeOptions(parentId int64, datasetList []model.SysDataset) []vo.Option {
	var result []vo.Option
	for _, dataset := range datasetList {
		if dataset.ParentID == parentId {
			option := vo.Option{
				Value: dataset.ID,
				Label: dataset.Name,
			}
			// 递归获取子数据集选项
			children := datasetService.recurDatasetTreeOptions(dataset.ID, datasetList)
			if len(children) > 0 {
				option.Children = children
			}
			result = append(result, option)
		}
	}
	return result
}

// GetDatasetForm 数据集表单数据
func (datasetService *DatasetService) GetDatasetForm(id int64) (datasetFormBO bo.DatasetFormBO, err error) {
	var dataset model.SysDataset
	err = global.DB.Model(&model.SysDataset{}).
		Where("id = ? AND deleted = ?", id, 0).
		First(&dataset).Error

	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return datasetFormBO, errors.New("数据集不存在")
		}
		return datasetFormBO, err
	}

	idPtr := dataset.ID
	datasetFormBO = bo.DatasetFormBO{
		ID:          &idPtr,
		ParentID:    dataset.ParentID,
		Type:        dataset.Type,
		Name:        dataset.Name,
		Description: dataset.Description,
		Path:        dataset.Path,
		Status:      dataset.Status,
	}

	return datasetFormBO, nil
}

// SaveDataset 新增数据集
func (datasetService *DatasetService) SaveDataset(datasetFormBO bo.DatasetFormBO) (err error) {
	// 创建数据集实体
	dataset := model.SysDataset{
		ParentID:    datasetFormBO.ParentID,
		Type:        datasetFormBO.Type,
		Name:        datasetFormBO.Name,
		Description: datasetFormBO.Description,
		Path:        datasetFormBO.Path,
		Status:      datasetFormBO.Status,
		Deleted:     0,
	}

	// 设置创建和更新时间
	dataset.CreatedAt = time.Now()
	dataset.UpdatedAt = time.Now()

	// 插入数据集
	err = global.DB.Create(&dataset).Error
	return err
}

// UpdateDataset 修改数据集
func (datasetService *DatasetService) UpdateDataset(id int64, datasetFormBO bo.DatasetFormBO) (err error) {
	// 更新数据集信息
	updates := map[string]interface{}{
		"parent_id":   datasetFormBO.ParentID,
		"type":        datasetFormBO.Type,
		"name":        datasetFormBO.Name,
		"description": datasetFormBO.Description,
		"path":        datasetFormBO.Path,
		"status":      datasetFormBO.Status,
		"update_time": time.Now(),
	}

	err = global.DB.Model(&model.SysDataset{}).
		Where("id = ? AND deleted = ?", id, 0).
		Updates(updates).Error

	return err
}

// DeleteDatasets 删除数据集
func (datasetService *DatasetService) DeleteDatasets(ids []int64) (err error) {
	if len(ids) == 0 {
		return errors.New("删除数据为空")
	}

	// 删除数据集（逻辑删除）
	err = global.DB.Model(&model.SysDataset{}).
		Where("id IN ?", ids).
		Updates(map[string]interface{}{
			"deleted":     1,
			"update_time": time.Now(),
		}).Error

	return err
}
