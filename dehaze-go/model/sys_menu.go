package model

// SysMenu 菜单管理
type SysMenu struct {
	BaseModel
	ParentID   int64     `gorm:"not null;comment:父菜单ID" json:"parentId"`
	TreePath   string    `gorm:"size:255;comment:父节点ID路径" json:"treePath"`
	Name       string    `gorm:"size:64;not null;default:'';comment:菜单名称" json:"name"`
	Type       int8      `gorm:"not null;comment:菜单类型(1:菜单 2:目录 3:外链 4:按钮)" json:"type"`
	Path       string    `gorm:"size:128;default:'';comment:路由路径(浏览器地址栏路径)" json:"path"`
	Component  string    `gorm:"size:128;comment:组件路径(vue页面完整路径，省略.vue后缀)" json:"component"`
	Perm       string    `gorm:"size:128;comment:权限标识" json:"perm"`
	Visible    int8      `gorm:"not null;default:1;comment:显示状态(1-显示;0-隐藏)" json:"visible"`
	Sort       int       `gorm:"type:int;default:0;comment:排序" json:"sort"`
	Icon       string    `gorm:"size:64;default:'';comment:菜单图标" json:"icon"`
	Redirect   string    `gorm:"size:128;comment:跳转路径" json:"redirect"`
	AlwaysShow int8      `gorm:"comment:【目录】只有一个子路由是否始终显示(1:是 0:否)" json:"alwaysShow"`
	KeepAlive  int8      `gorm:"comment:【菜单】是否开启页面缓存(1:是 0:否)" json:"keepAlive"`
	Roles      []SysRole `gorm:"many2many:sys_role_menu;joinForeignKey:menu_id;joinReferences:role_id"`
}
