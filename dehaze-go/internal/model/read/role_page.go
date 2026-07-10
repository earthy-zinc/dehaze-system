package read

import "time"

// RolePage 角色分页读模型
type RolePage struct {
	ID            int64     `json:"id"`
	Name          string    `json:"name"`
	Code          string    `json:"code"`
	DataScope     int8      `json:"dataScope"`
	DataScopeLabel string   `json:"dataScopeLabel"`
	Status        int       `json:"status"`
	Sort          int       `json:"sort"`
	CreateTime    time.Time `json:"createTime"`
	UpdateTime    time.Time `json:"updateTime"`
}
