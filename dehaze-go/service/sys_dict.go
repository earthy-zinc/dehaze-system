package service

import (
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/model/vo"
	"gorm.io/gorm"
)

type DictService struct{}

// GetDictPage 字典分页列表
func (dictService *DictService) GetDictPage(queryParams query.DictPageQuery) (result vo.PageResult[vo.DictPageVO], err error) {
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
	db := global.DB.Model(&model.SysDict{})

	// 添加查询条件
	if queryParams.Keywords != "" {
		keyword := "%" + queryParams.Keywords + "%"
		db = db.Where("name LIKE ?", keyword)
	}
	if queryParams.TypeCode != "" {
		db = db.Where("type_code = ?", queryParams.TypeCode)
	}

	// 查询总数
	var total int64
	err = db.Count(&total).Error
	if err != nil {
		return result, err
	}

	// 分页查询
	var dictList []model.SysDict
	err = db.Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&dictList).Error
	if err != nil {
		return result, err
	}

	// 转换为VO
	var dictPageVOs []vo.DictPageVO
	for _, dict := range dictList {
		dictPageVO := vo.DictPageVO{
			ID:     dict.ID,
			Name:   dict.Name,
			Value:  dict.Value,
			Status: dict.Status,
		}
		dictPageVOs = append(dictPageVOs, dictPageVO)
	}

	// 构造分页结果
	result.List = dictPageVOs
	result.Total = total
	result.PageNum = pageNum
	result.PageSize = pageSize

	return result, nil
}

// GetDictForm 字典数据表单数据
func (dictService *DictService) GetDictForm(id int64) (dictFormBO bo.DictFormBO, err error) {
	var dict model.SysDict
	err = global.DB.Model(&model.SysDict{}).
		Where("id = ?", id).
		Select("id, type_code, name, value, status, sort, remark").
		First(&dict).Error

	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return dictFormBO, errors.New("字典数据项不存在")
		}
		return dictFormBO, err
	}

	idPtr := dict.ID
	dictFormBO = bo.DictFormBO{
		ID:       &idPtr,
		TypeCode: dict.TypeCode,
		Name:     dict.Name,
		Value:    dict.Value,
		Status:   dict.Status,
		Sort:     dict.Sort,
		Remark:   dict.Remark,
	}

	return dictFormBO, nil
}

// SaveDict 新增字典
func (dictService *DictService) SaveDict(dictFormBO bo.DictFormBO) (err error) {
	dict := model.SysDict{
		TypeCode:  dictFormBO.TypeCode,
		Name:      dictFormBO.Name,
		Value:     dictFormBO.Value,
		Status:    dictFormBO.Status,
		Sort:      dictFormBO.Sort,
		Remark:    dictFormBO.Remark,
		Defaulted: 0, // 默认为0
	}

	// 设置创建和更新时间
	dict.CreatedAt = time.Now()
	dict.UpdatedAt = time.Now()

	// 插入字典数据
	err = global.DB.Create(&dict).Error
	return err
}

// UpdateDict 修改字典
func (dictService *DictService) UpdateDict(id int64, dictFormBO bo.DictFormBO) (err error) {
	// 更新字典信息
	updates := map[string]interface{}{
		"type_code":  dictFormBO.TypeCode,
		"name":       dictFormBO.Name,
		"value":      dictFormBO.Value,
		"status":     dictFormBO.Status,
		"sort":       dictFormBO.Sort,
		"remark":     dictFormBO.Remark,
		"updated_at": time.Now(),
	}

	err = global.DB.Model(&model.SysDict{}).
		Where("id = ?", id).
		Updates(updates).Error

	return err
}

// DeleteDict 删除字典
func (dictService *DictService) DeleteDict(ids string) (err error) {
	if ids == "" {
		return errors.New("删除数据为空")
	}

	// 解析ID列表
	idStrings := strings.Split(ids, ",")
	var idList []int64
	for _, idStr := range idStrings {
		id, err := strconv.ParseInt(idStr, 10, 64)
		if err != nil {
			return errors.New("字典数据项ID格式不正确")
		}
		idList = append(idList, id)
	}

	// 删除字典数据项
	err = global.DB.Where("id IN ?", idList).Delete(&model.SysDict{}).Error
	return err
}

// ListDictOptions 获取字典下拉列表
func (dictService *DictService) ListDictOptions(typeCode string) (options []vo.Option, err error) {
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