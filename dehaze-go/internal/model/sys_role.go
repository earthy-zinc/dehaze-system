package model

// SysRole 角色表
type SysRole struct {
	BaseModel
	Name      string    `gorm:"column:name;type:varchar(64);not null;uniqueIndex:idx_sys_role_name;comment:角色名称" json:"name"`
	Code      string    `gorm:"column:code;type:varchar(32);comment:角色编码" json:"code"`
	Sort      int       `gorm:"column:sort;type:int;comment:显示顺序" json:"sort"`
	Status    int8      `gorm:"column:status;type:tinyint(1);default:1;comment:角色状态(1-正常；0-停用)" json:"status"`
	DataScope int8      `gorm:"column:data_scope;type:tinyint;comment:数据权限(0-所有数据；1-部门及子部门数据；2-本部门数据；3-本人数据)" json:"dataScope"`
	Deleted   int8      `gorm:"column:deleted;type:tinyint(1);not null;default:0;comment:逻辑删除标识(0-未删除；1-已删除)" json:"deleted"`
	Users     []SysUser `gorm:"many2many:sys_user_role;joinForeignKey:role_id;joinReferences:user_id"`
	Menus     []SysMenu `gorm:"many2many:sys_role_menu;joinForeignKey:role_id;joinReferences:menu_id"`
}
