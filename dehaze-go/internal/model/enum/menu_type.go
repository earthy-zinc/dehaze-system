package enum

// MenuType 菜单类型枚举
// 注意：值对齐 Java MenuTypeEnum 与数据库 sys_menu.type 实际数据
// （系统管理 type=2 为目录，用户管理 type=1 为菜单）
const (
	// MenuTypeMenu 菜单类型 - 对应具体的页面功能，可配置路由和组件
	MenuTypeMenu = 1
	// MenuTypeCatalog 目录类型 - 用于组织菜单结构，本身不对应具体页面
	MenuTypeCatalog = 2
	// MenuTypeExtlink 外链类型 - 外部链接，跳转到外部系统或文档
	MenuTypeExtlink = 3
	// MenuTypeButton 按钮类型 - 页面内的操作按钮，配置权限标识供前端权限控制使用
	MenuTypeButton = 4
)

// MenuTypeNames 菜单类型名称映射
var MenuTypeNames = map[int]string{
	MenuTypeMenu:    "菜单",
	MenuTypeCatalog: "目录",
	MenuTypeExtlink: "外链",
	MenuTypeButton:  "按钮",
}

// MenuTypeEnumNames 整数 → 字符串枚举名映射（用于响应序列化，对齐 Java MenuTypeEnum 的 Jackson 序列化）
var MenuTypeEnumNames = map[int]string{
	MenuTypeMenu:    "MENU",
	MenuTypeCatalog: "CATALOG",
	MenuTypeExtlink: "EXTLINK",
	MenuTypeButton:  "BUTTON",
}

// MenuTypeNameToValue 字符串枚举名 → 整数映射（用于请求反序列化）
var MenuTypeNameToValue = map[string]int{
	"MENU":    MenuTypeMenu,
	"CATALOG": MenuTypeCatalog,
	"EXTLINK": MenuTypeExtlink,
	"BUTTON":  MenuTypeButton,
}

// GetMenuTypeName 获取菜单类型名称（中文）
func GetMenuTypeName(menuType int) string {
	if name, ok := MenuTypeNames[menuType]; ok {
		return name
	}
	return "未知"
}

// GetMenuTypeEnumName 获取菜单类型枚举名（英文，用于 JSON 序列化）
func GetMenuTypeEnumName(menuType int) string {
	if name, ok := MenuTypeEnumNames[menuType]; ok {
		return name
	}
	return ""
}
