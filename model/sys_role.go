package model

// SysRole 角色表
type SysRole struct {
	BASE_MODEL
	Name      string    `gorm:"size:64;not null;uniqueIndex;comment:角色名称" json:"name"`
	Code      string    `gorm:"size:32;comment:角色编码" json:"code"`
	Sort      int       `gorm:"comment:显示顺序" json:"sort"`
	Status    int8      `gorm:"default:1;comment:角色状态(1-正常；0-停用)" json:"status"`
	DataScope int8      `gorm:"comment:数据权限(0-所有数据；1-部门及子部门数据；2-本部门数据；3-本人数据)" json:"dataScope"`
	Deleted   int8      `gorm:"not null;default:0;comment:逻辑删除标识(0-未删除；1-已删除)" json:"deleted"`
	Users     []SysUser `gorm:"many2many:sys_user_role;joinForeignKey:role_id;joinReferences:user_id"`
	Menus     []SysMenu `gorm:"many2many:sys_role_menu;joinForeignKey:role_id;joinReferences:menu_id"`
}
