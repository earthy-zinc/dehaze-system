package bo

// MenuForm 菜单表单对象
type MenuForm struct {
	// 菜单ID
	ID *int64 `json:"id"`
	// 父菜单ID
	ParentID int64 `json:"parentId"`
	// 菜单名称
	Name string `json:"name"`
	// 菜单类型(1-菜单；2-目录；3-外链；4-按钮权限)
	Type int8 `json:"type"`
	// 路由路径
	Path string `json:"path"`
	// 组件路径(vue页面完整路径，省略.vue后缀)
	Component string `json:"component"`
	// 权限标识
	Perm string `json:"perm"`
	// 显示状态(1:显示;0:隐藏)
	Visible int `json:"visible"`
	// 排序(数字越小排名越靠前)
	Sort int `json:"sort"`
	// 菜单图标
	Icon string `json:"icon"`
	// 跳转路径
	Redirect string `json:"redirect"`
	// 【菜单】是否开启页面缓存(1:开启;0:关闭)
	KeepAlive int `json:"keepAlive"`
	// 【目录】只有一个子路由是否始终显示(1:是 0:否)
	AlwaysShow int `json:"alwaysShow"`
}