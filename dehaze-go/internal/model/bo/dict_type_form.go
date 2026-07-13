package bo

// DictTypeFormBO 字典类型表单对象
type DictTypeFormBO struct {
	// 字典类型ID
	ID *int64 `json:"id"`
	// 类型名称
	Name string `json:"name" binding:"required,max=50,no_xss"`
	// 类型编码
	Code string `json:"code" binding:"required,max=50,no_xss"`
	// 类型状态(1:启用;0:禁用)
	Status int8 `json:"status" binding:"oneof=0 1"`
	// 备注
	Remark string `json:"remark" binding:"max=255"`
}
