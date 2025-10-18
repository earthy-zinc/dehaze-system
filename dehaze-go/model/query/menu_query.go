package query

// MenuQuery 菜单查询对象
type MenuQuery struct {
	// 关键字(菜单名称)
	Keywords string `json:"keywords"`
	// 状态(1->显示；0->隐藏)
	Status *int `json:"status"`
}