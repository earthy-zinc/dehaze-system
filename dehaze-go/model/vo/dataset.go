package vo

import "time"

// DatasetVO 数据集视图对象
type DatasetVO struct {
	// 数据集ID
	ID int64 `json:"id"`
	// 父数据集ID
	ParentID int64 `json:"parentId"`
	// 数据集类型
	Type string `json:"type"`
	// 数据集名称
	Name string `json:"name"`
	// 数据集描述
	Description string `json:"description"`
	// 存储位置
	Path string `json:"path"`
	// 占用空间大小
	Size string `json:"size"`
	// 子数据集
	Children []DatasetVO `json:"children"`
	// 创建时间
	CreateTime time.Time `json:"createTime"`
	// 修改时间
	UpdateTime time.Time `json:"updateTime"`
	// 状态(1:启用；0:禁用)
	Status int `json:"status"`
}
