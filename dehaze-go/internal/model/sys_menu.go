package model

// SysMenu 菜单管理
type SysMenu struct {
	BaseModel
	ParentID   int64     `gorm:"column:parent_id;type:bigint;not null;comment:父菜单ID" json:"parentId"`
	TreePath   string    `gorm:"column:tree_path;type:varchar(255);comment:父节点ID路径" json:"treePath"`
	Name       string    `gorm:"column:name;type:varchar(64);not null;default:'';comment:菜单名称" json:"name"`
	Type       int8      `gorm:"column:type;type:tinyint;not null;comment:菜单类型(1:菜单 2:目录 3:外链 4:按钮)" json:"type"`
	Path       string    `gorm:"column:path;type:varchar(128);default:'';comment:路由路径(浏览器地址栏路径)" json:"path"`
	Component  string    `gorm:"column:component;type:varchar(128);comment:组件路径(vue页面完整路径，省略.vue后缀)" json:"component"`
	Perm       string    `gorm:"column:perm;type:varchar(128);comment:权限标识" json:"perm"`
	Visible    int8      `gorm:"column:visible;type:tinyint;not null;default:1;comment:显示状态(1-显示;0-隐藏)" json:"visible"`
	Status     int8      `gorm:"column:status;type:tinyint;not null;default:1;comment:状态(1:启用;0:禁用)" json:"status"`
	Sort       int       `gorm:"column:sort;type:int;default:0;comment:排序" json:"sort"`
	Icon       string    `gorm:"column:icon;type:varchar(64);default:'';comment:菜单图标" json:"icon"`
	Redirect   string    `gorm:"column:redirect;type:varchar(128);comment:跳转路径" json:"redirect"`
	AlwaysShow int8      `gorm:"column:always_show;type:tinyint;comment:【目录】只有一个子路由是否始终显示(1:是 0:否)" json:"alwaysShow"`
	KeepAlive  int8      `gorm:"column:keep_alive;type:tinyint;comment:【菜单】是否开启页面缓存(1:是 0:否)" json:"keepAlive"`
	Deleted    int8      `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识(0:未删除;1:已删除)" json:"deleted"`
	Roles      []SysRole `gorm:"many2many:sys_role_menu;joinForeignKey:menu_id;joinReferences:role_id"`
}
