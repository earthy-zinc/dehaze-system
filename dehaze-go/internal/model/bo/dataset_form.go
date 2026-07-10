package bo

import (
	"encoding/json"
	"fmt"
)

// DatasetFormBO 数据集表单业务对象
type DatasetFormBO struct {
	// 数据集ID
	ID *int64 `json:"id"`
	// 父数据集ID
	ParentID int64 `json:"parentId" binding:"required"`
	// 数据集类型
	Type string `json:"type"`
	// 数据集名称
	Name string `json:"name" binding:"required,max=128"`
	// 数据集描述
	Description string `json:"description" binding:"omitempty,max=255"`
	// 数据集存储路径
	Path string `json:"path"`
	// 状态(1:正常;0:禁用) - 支持字符串和数字类型
	Status int8 `json:"status"`
	// 创建时间
	CreateTime string `json:"createTime,omitempty"`
	// 更新时间
	UpdateTime string `json:"updateTime,omitempty"`
	// 统计信息
	Statistics *StatisticsBO `json:"statistics"`
}

// StatisticsBO 数据集统计信息
type StatisticsBO struct {
	ItemCount          int64            `json:"itemCount"`
	FileCount          int64            `json:"fileCount"`
	TotalSize          int64            `json:"totalSize"`
	ClearCount         int64            `json:"clearCount"`
	HazyCount          int64            `json:"hazyCount"`
	SceneDistribution  map[string]int64 `json:"sceneDistribution"`
	HazeDistribution   map[string]int64 `json:"hazeDistribution"`
	FormatDistribution map[string]int64 `json:"formatDistribution"`
}

// UnmarshalJSON 自定义JSON解析，支持字符串到int8的转换
func (d *DatasetFormBO) UnmarshalJSON(data []byte) error {
	// 定义临时类型避免递归调用 UnmarshalJSON
	type Alias DatasetFormBO
	aux := &struct {
		Status interface{} `json:"status"`
		*Alias
	}{
		Alias: (*Alias)(d),
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	// 处理 Status 字段，支持字符串和数字
	switch v := aux.Status.(type) {
	case float64: // JSON 数字默认解析为 float64
		d.Status = int8(v)
	case string:
		var status int8
		if _, err := fmt.Sscanf(v, "%d", &status); err != nil {
			return fmt.Errorf("invalid status format: %s", v)
		}
		d.Status = status
	case nil:
		// 如果为空，使用默认值 1
		d.Status = 1
	}

	return nil
}
