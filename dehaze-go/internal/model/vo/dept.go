package vo

import "time"

// DeptVO 部门视图对象
type DeptVO struct {
	// 部门ID
	ID int64 `json:"id"`
	// 父部门ID
	ParentID int64 `json:"parentId"`
	// 部门名称
	Name string `json:"name"`
	// 排序
	Sort int `json:"sort"`
	// 状态(1:启用；0:禁用)
	Status int8 `json:"status"`
	// 子部门
	Children []DeptVO `json:"children,omitempty"`
	// 创建时间
	CreateTime time.Time `json:"createTime"`
	// 修改时间
	UpdateTime time.Time `json:"updateTime"`
}
