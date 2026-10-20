package vo

import "time"

// DictPageVO 字典分页对象
type DictPageVO struct {
	// 字典ID
	ID int64 `json:"id"`
	// 字典名称
	Name string `json:"name"`
	// 字典值
	Value string `json:"value"`
	// 类型编码
	TypeCode string `json:"typeCode"`
	// 是否默认(1:是;0:否)
	Defaulted int8 `json:"defaulted"`
	// 排序
	Sort int `json:"sort"`
	// 状态(1:启用;0:禁用)
	Status int8 `json:"status"`
	// 备注
	Remark string `json:"remark"`
	// 创建时间
	CreateTime time.Time `json:"createTime"`
}
