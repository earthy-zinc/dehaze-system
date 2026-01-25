package model

// SysRoleMenu 角色和菜单关联表
type SysRoleMenu struct {
	RoleID int64 `gorm:"column:role_id;type:bigint;primaryKey;comment:角色ID" json:"roleId"`
	MenuID int64 `gorm:"column:menu_id;type:bigint;primaryKey;comment:菜单ID" json:"menuId"`
}

func (SysRoleMenu) TableName() string {
	return "sys_role_menu"
}
