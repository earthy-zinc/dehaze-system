package model

// SysUserRole 用户和角色关联表
type SysUserRole struct {
	UserID int64 `gorm:"column:user_id;type:bigint;primaryKey;comment:用户ID" json:"userId"`
	RoleID int64 `gorm:"column:role_id;type:bigint;primaryKey;comment:角色ID" json:"roleId"`
}

func (SysUserRole) TableName() string {
	return "sys_user_role"
}
