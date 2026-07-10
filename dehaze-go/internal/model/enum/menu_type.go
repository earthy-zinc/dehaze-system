package enum

// MenuType 菜单类型枚举
const (
	// MenuTypeCatalog 目录类型 - 用于组织菜单结构，本身不对应具体页面
	MenuTypeCatalog = 1
	// MenuTypeMenu 菜单类型 - 对应具体的页面功能，可配置路由和组件
	MenuTypeMenu = 2
	// MenuTypeExtlink 外链类型 - 外部链接，跳转到外部系统或文档
	MenuTypeExtlink = 3
	// MenuTypeButton 按钮类型 - 页面内的操作按钮，配置权限标识供前端权限控制使用
	MenuTypeButton = 4
)

// MenuTypeNames 菜单类型名称映射
var MenuTypeNames = map[int]string{
	MenuTypeCatalog: "目录",
	MenuTypeMenu:    "菜单",
	MenuTypeExtlink: "外链",
	MenuTypeButton:  "按钮",
}

// GetMenuTypeName 获取菜单类型名称
func GetMenuTypeName(menuType int) string {
	if name, ok := MenuTypeNames[menuType]; ok {
		return name
	}
	return "未知"
}
