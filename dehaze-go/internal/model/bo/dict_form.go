package bo

// DictFormBO 字典表单对象
type DictFormBO struct {
	// 字典ID
	ID *int64 `json:"id"`
	// 类型编码
	TypeCode string `json:"typeCode" binding:"required,max=32"`
	// 字典名称
	Name string `json:"name" binding:"required,max=50,no_xss"`
	// 字典值
	Value string `json:"value" binding:"required,max=50,no_xss"`
	// 状态(1:启用;0:禁用)
	Status int8 `json:"status" binding:"oneof=0 1"`
	// 是否默认(1:是;0:否)
	Defaulted int8 `json:"defaulted" binding:"oneof=0 1"`
	// 排序
	Sort int `json:"sort" binding:"min=1"`
	// 字典备注
	Remark string `json:"remark" binding:"max=255"`
}
