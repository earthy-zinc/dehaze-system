package service

import (
	"errors"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/model/vo"
	"gorm.io/gorm"
)

type DictTypeService struct {
	dictService DictService
}

// GetDictTypePage 字典类型分页列表
func (dictTypeService *DictTypeService) GetDictTypePage(queryParams query.DictTypePageQuery) (result vo.PageResult[vo.DictTypePageVO], err error) {
	// 初始化分页参数
	pageNum := queryParams.PageNum
	pageSize := queryParams.PageSize
	if pageNum <= 0 {
		pageNum = 1
	}
	if pageSize <= 0 {
		pageSize = 10
	}

	// 构建查询
	db := global.DB.Model(&model.SysDictType{})

	// 添加查询条件
	if queryParams.Keywords != "" {
		keyword := "%" + queryParams.Keywords + "%"
		db = db.Where("name LIKE ? OR code LIKE ?", keyword, keyword)
	}

	// 查询总数
	var total int64
	err = db.Count(&total).Error
	if err != nil {
		return result, err
	}

	// 分页查询
	var dictTypeList []model.SysDictType
	err = db.Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&dictTypeList).Error
	if err != nil {
		return result, err
	}

	// 转换为VO
	var dictTypePageVOs []vo.DictTypePageVO
	for _, dictType := range dictTypeList {
		dictTypePageVO := vo.DictTypePageVO{
			ID:     dictType.ID,
			Name:   dictType.Name,
			Code:   dictType.Code,
			Status: dictType.Status,
		}
		dictTypePageVOs = append(dictTypePageVOs, dictTypePageVO)
	}

	// 构造分页结果
	result.List = dictTypePageVOs
	result.Total = total
	result.PageNum = pageNum
	result.PageSize = pageSize

	return result, nil
}

// GetDictTypeForm 字典类型表单数据
func (dictTypeService *DictTypeService) GetDictTypeForm(id int64) (dictTypeFormBO bo.DictTypeFormBO, err error) {
	var dictType model.SysDictType
	err = global.DB.Model(&model.SysDictType{}).
		Where("id = ?", id).
		Select("id, name, code, status, remark").
		First(&dictType).Error

	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return dictTypeFormBO, errors.New("字典类型不存在")
		}
		return dictTypeFormBO, err
	}

	idPtr := dictType.ID
	dictTypeFormBO = bo.DictTypeFormBO{
		ID:     &idPtr,
		Name:   dictType.Name,
		Code:   dictType.Code,
		Status: dictType.Status,
		Remark: dictType.Remark,
	}

	return dictTypeFormBO, nil
}

// SaveDictType 新增字典类型
func (dictTypeService *DictTypeService) SaveDictType(dictTypeFormBO bo.DictTypeFormBO) (err error) {
	dictType := model.SysDictType{
		Name:   dictTypeFormBO.Name,
		Code:   dictTypeFormBO.Code,
		Status: dictTypeFormBO.Status,
		Remark: dictTypeFormBO.Remark,
	}

	// 设置创建和更新时间
	dictType.CreatedAt = time.Now()
	dictType.UpdatedAt = time.Now()

	// 插入字典类型
	err = global.DB.Create(&dictType).Error
	return err
}

// UpdateDictType 修改字典类型
func (dictTypeService *DictTypeService) UpdateDictType(id int64, dictTypeFormBO bo.DictTypeFormBO) (err error) {
	// 获取字典类型
	var oldDictType model.SysDictType
	err = global.DB.Where("id = ?", id).First(&oldDictType).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return errors.New("字典类型不存在")
		}
		return err
	}

	// 更新字典类型信息
	updates := map[string]interface{}{
		"name":       dictTypeFormBO.Name,
		"code":       dictTypeFormBO.Code,
		"status":     dictTypeFormBO.Status,
		"remark":     dictTypeFormBO.Remark,
		"updated_at": time.Now(),
	}

	err = global.DB.Model(&model.SysDictType{}).
		Where("id = ?", id).
		Updates(updates).Error

	if err != nil {
		return err
	}

	// 字典类型code变化，同步修改字典项的类型code
	if oldDictType.Code != dictTypeFormBO.Code {
		err = global.DB.Model(&model.SysDict{}).
			Where("type_code = ?", oldDictType.Code).
			Updates(map[string]interface{}{
				"type_code": dictTypeFormBO.Code,
			}).Error
	}

	return err
}

// DeleteDictTypes 删除字典类型
func (dictTypeService *DictTypeService) DeleteDictTypes(ids string) (err error) {
	if ids == "" {
		return errors.New("删除数据为空")
	}

	// 解析ID列表
	idStrings := strings.Split(ids, ",")
	var idList []string
	for _, idStr := range idStrings {
		idList = append(idList, idStr)
	}

	// 获取要删除的字典类型编码
	var dictTypes []model.SysDictType
	err = global.DB.Model(&model.SysDictType{}).
		Where("id IN ?", idList).
		Select("code").
		Find(&dictTypes).Error

	if err != nil {
		return err
	}

	// 删除字典数据项
	var dictTypeCodes []string
	for _, dictType := range dictTypes {
		dictTypeCodes = append(dictTypeCodes, dictType.Code)
	}

	if len(dictTypeCodes) > 0 {
		err = global.DB.Where("type_code IN ?", dictTypeCodes).Delete(&model.SysDict{}).Error
		if err != nil {
			return err
		}
	}

	// 删除字典类型
	err = global.DB.Where("id IN ?", idList).Delete(&model.SysDictType{}).Error
	return err
}

// ListDictItemsByTypeCode 获取字典类型的数据项
func (dictTypeService *DictTypeService) ListDictItemsByTypeCode(typeCode string) (options []vo.Option, err error) {
	// 查询字典数据项
	var dictList []model.SysDict
	err = global.DB.Model(&model.SysDict{}).
		Where("type_code = ?", typeCode).
		Select("value, name").
		Find(&dictList).Error

	if err != nil {
		return options, err
	}

	// 转换下拉数据
	for _, dict := range dictList {
		option := vo.Option{
			Value: dict.Value,
			Label: dict.Name,
		}
		options = append(options, option)
	}

	return options, nil
}
