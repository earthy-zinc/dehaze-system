package service

import (
	"errors"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/model/vo"
	"gorm.io/gorm"
)

type AlgorithmService struct{}

// GetAlgorithmList 获取算法树形列表
func (algorithmService *AlgorithmService) GetAlgorithmList(queryParams query.AlgorithmQuery) (algorithmVOs []vo.AlgorithmVO, err error) {
	// 构建查询
	db := global.DB.Model(&model.SysAlgorithm{}).
		Where("status = ?", 1) // 只查询启用的算法

	// 添加查询条件
	if queryParams.Keywords != "" {
		keyword := "%" + queryParams.Keywords + "%"
		db = db.Where("name LIKE ?", keyword)
	}

	// 查询数据
	var algorithmList []model.SysAlgorithm
	err = db.Find(&algorithmList).Error
	if err != nil {
		return algorithmVOs, err
	}

	if len(algorithmList) == 0 {
		return algorithmVOs, nil
	}

	// 构建算法树
	algorithmVOs = buildAlgorithmTree(0, algorithmList)
	return algorithmVOs, nil
}

// GetAlgorithmOptions 获取算法下拉选项
func (algorithmService *AlgorithmService) GetAlgorithmOptions() (options []vo.Option, err error) {
	var algorithms []model.SysAlgorithm
	err = global.DB.Model(&model.SysAlgorithm{}).
		Where("status = ?", 1).
		Select("id, name").
		Find(&algorithms).Error
	if err != nil {
		return options, err
	}

	options = make([]vo.Option, len(algorithms))
	for i, algorithm := range algorithms {
		options[i] = vo.Option{
			Value: algorithm.ID,
			Label: algorithm.Name,
		}
	}
	return options, nil
}

// GetAlgorithmById 根据ID获取算法
func (algorithmService *AlgorithmService) GetAlgorithmById(id int64) (algorithm *model.SysAlgorithm, err error) {
	algorithm = &model.SysAlgorithm{}
	err = global.DB.Where("id = ? AND status = ?", id, 1).First(algorithm).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, errors.New("算法不存在")
		}
		return nil, err
	}
	return algorithm, nil
}

// GetRootAlgorithm 获取根算法
func (algorithmService *AlgorithmService) GetRootAlgorithm(id int64) (algorithm *model.SysAlgorithm, err error) {
	algorithm, err = algorithmService.GetAlgorithmById(id)
	if err != nil {
		return nil, err
	}

	// 循环查找根节点
	for algorithm.ParentID != 0 {
		algorithm, err = algorithmService.GetAlgorithmById(algorithm.ParentID)
		if err != nil {
			return nil, errors.New("无法获取算法根节点")
		}
	}
	return algorithm, nil
}

// AddAlgorithm 添加算法
func (algorithmService *AlgorithmService) AddAlgorithm(algorithmForm bo.AlgorithmFormBO) (err error) {
	algorithm := model.SysAlgorithm{
		ParentID:    algorithmForm.ParentID,
		Type:        algorithmForm.Type,
		Name:        algorithmForm.Name,
		Path:        algorithmForm.Path,
		ImportPath:  algorithmForm.ImportPath,
		Description: algorithmForm.Description,
		Status:      int8(algorithmForm.Status),
	}

	// 如果父节点ID不为0，检查父节点是否存在
	if algorithm.ParentID != 0 {
		var parentAlgorithm model.SysAlgorithm
		err = global.DB.First(&parentAlgorithm, algorithm.ParentID).Error
		if err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return errors.New("父算法不存在")
			}
			return err
		}
	}

	// 创建算法
	err = global.DB.Create(&algorithm).Error
	return err
}

// UpdateAlgorithm 更新算法
func (algorithmService *AlgorithmService) UpdateAlgorithm(id int64, algorithmForm bo.AlgorithmFormBO) (err error) {
	// 检查算法是否存在
	var algorithm model.SysAlgorithm
	err = global.DB.First(&algorithm, id).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return errors.New("算法不存在")
		}
		return err
	}

	// 更新算法信息
	algorithm.ParentID = algorithmForm.ParentID
	algorithm.Type = algorithmForm.Type
	algorithm.Name = algorithmForm.Name
	algorithm.Path = algorithmForm.Path
	algorithm.ImportPath = algorithmForm.ImportPath
	algorithm.Description = algorithmForm.Description
	algorithm.Status = int8(algorithmForm.Status)

	// 如果父节点ID不为0，检查父节点是否存在
	if algorithm.ParentID != 0 && algorithm.ParentID != id {
		var parentAlgorithm model.SysAlgorithm
		err = global.DB.First(&parentAlgorithm, algorithm.ParentID).Error
		if err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return errors.New("父算法不存在")
			}
			return err
		}
	}

	// 更新算法
	err = global.DB.Save(&algorithm).Error
	return err
}

// DeleteAlgorithms 删除算法
func (algorithmService *AlgorithmService) DeleteAlgorithms(ids []int64) (err error) {
	if len(ids) == 0 {
		return errors.New("请选择要删除的算法")
	}

	// 检查是否有子算法
	var count int64
	err = global.DB.Model(&model.SysAlgorithm{}).
		Where("parent_id IN ?", ids).
		Count(&count).Error
	if err != nil {
		return err
	}

	if count > 0 {
		return errors.New("存在子算法，无法删除")
	}

	// 删除算法
	err = global.DB.Delete(&model.SysAlgorithm{}, ids).Error
	return err
}

// buildAlgorithmTree 构建算法树
func buildAlgorithmTree(parentId int64, algorithms []model.SysAlgorithm) []vo.AlgorithmVO {
	var tree []vo.AlgorithmVO

	for _, algorithm := range algorithms {
		if algorithm.ParentID == parentId {
			node := vo.AlgorithmVO{
				ID:          algorithm.ID,
				Name:        algorithm.Name,
				Type:        algorithm.Type,
				Img:         algorithm.Img,
				Description: algorithm.Description,
				Path:        algorithm.Path,
				Flops:       algorithm.Flops,
				Params:      algorithm.Params,
				ImportPath:  algorithm.ImportPath,
				Status:      int(algorithm.Status),
				Size:        algorithm.Size,
			}
			// 递归构建子树
			node.Children = buildAlgorithmTree(algorithm.ID, algorithms)
			tree = append(tree, node)
		}
	}

	return tree
}
