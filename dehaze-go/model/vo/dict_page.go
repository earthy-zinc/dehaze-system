package vo

// DictPageVO 字典分页对象
type DictPageVO struct {
	// 字典ID
	ID int64 `json:"id"`
	// 字典名称
	Name string `json:"name"`
	// 字典值
	Value string `json:"value"`
	// 状态(1:启用;0:禁用)
	Status int8 `json:"status"`
}