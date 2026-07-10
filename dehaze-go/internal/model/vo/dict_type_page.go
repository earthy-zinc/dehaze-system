package vo

import "time"

// DictTypePageVO 字典类型分页对象
type DictTypePageVO struct {
	// 字典类型ID
	ID int64 `json:"id"`
	// 类型名称
	Name string `json:"name"`
	// 类型编码
	Code string `json:"code"`
	// 状态：1:启用;0:禁用
	Status int8 `json:"status"`
	// 备注
	Remark string `json:"remark"`
	// 创建时间
	CreateTime time.Time `json:"createTime"`
}
