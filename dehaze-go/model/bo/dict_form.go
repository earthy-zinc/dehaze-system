package bo

// DictFormBO 字典表单对象
type DictFormBO struct {
	// 字典ID
	ID *int64 `json:"id"`
	// 类型编码
	TypeCode string `json:"typeCode"`
	// 字典名称
	Name string `json:"name"`
	// 字典值
	Value string `json:"value"`
	// 状态(1:启用;0:禁用)
	Status int8 `json:"status"`
	// 排序
	Sort int `json:"sort"`
	// 字典备注
	Remark string `json:"remark"`
}
