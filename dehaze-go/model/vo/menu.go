package vo

// MenuVO 菜单视图对象
type MenuVO struct {
	// 菜单ID
	ID int64 `json:"id"`
	// 父菜单ID
	ParentID int64 `json:"parentId"`
	// 菜单名称
	Name string `json:"name"`
	// 路由路径
	Path string `json:"path"`
	// 组件路径
	Component string `json:"component"`
	// 菜单排序(数字越小排名越靠前)
	Sort int `json:"sort"`
	// 菜单是否可见(1:显示;0:隐藏)
	Visible int `json:"visible"`
	// ICON
	Icon string `json:"icon"`
	// 跳转路径
	Redirect string `json:"redirect"`
	// 按钮权限标识
	Perm string `json:"perm"`
	// 子菜单
	Children []MenuVO `json:"children"`
}