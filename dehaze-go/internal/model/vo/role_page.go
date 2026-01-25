package vo

import "time"

// RolePageVO 角色分页对象
type RolePageVO struct {
	// 角色ID
	ID int64 `json:"id"`
	// 角色名称
	Name string `json:"name"`
	// 角色编码
	Code string `json:"code"`
	// 数据权限范围
	DataScope int8 `json:"dataScope"`
	// 数据权限范围中文描述
	DataScopeLabel string `json:"dataScopeLabel"`
	// 角色状态
	Status int `json:"status"`
	// 排序
	Sort int `json:"sort"`
	// 创建时间
	CreateTime time.Time `json:"createTime"`
	// 修改时间
	UpdateTime time.Time `json:"updateTime"`
}
