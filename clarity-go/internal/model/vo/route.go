package vo

// RouteVO 菜单路由视图对象
type RouteVO struct {
	// 路由路径
	Path string `json:"path"`
	// 组件路径
	Component string `json:"component"`
	// 跳转链接
	Redirect string `json:"redirect"`
	// 路由名称
	Name string `json:"name"`
	// 路由属性
	Meta RouteMeta `json:"meta"`
	// 子路由列表
	Children []RouteVO `json:"children"`
}

// RouteMeta 路由属性类型
type RouteMeta struct {
	// 路由title
	Title string `json:"title"`
	// ICON
	Icon string `json:"icon"`
	// 是否隐藏(true-是 false-否)
	Hidden bool `json:"hidden"`
	// 拥有路由权限的角色编码
	Roles []string `json:"roles"`
	// 【菜单】是否开启页面缓存
	KeepAlive *bool `json:"keepAlive"`
	// 【目录】只有一个子路由是否始终显示
	AlwaysShow *bool `json:"alwaysShow"`
}
