package query

// MenuQuery 菜单查询对象
type MenuQuery struct {
	// 关键字(菜单名称)
	Keywords string `json:"keywords"`
	// 权限标识（模糊筛选）
	Perm string `json:"perm"`
	// 路由地址（模糊筛选）
	Path string `json:"path"`
	// 菜单类型
	Type *int `json:"type"`
	// 显示状态(1:显示;0:隐藏)
	Visible *int `json:"visible"`
}
