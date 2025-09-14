package model

// SysUser 用户信息表
type SysUser struct {
	BASE_MODEL
	Username string    `gorm:"size:64;uniqueIndex;comment:用户名" json:"username"`
	Nickname string    `gorm:"size:64;comment:昵称" json:"nickname"`
	Gender   int8      `gorm:"default:1;comment:性别((1:男;2:女))" json:"gender"`
	Password string    `gorm:"size:100;comment:密码" json:"password"`
	DeptID   int       `gorm:"comment:部门ID" json:"deptId"`
	Avatar   string    `gorm:"type:text;comment:用户头像" json:"avatar"`
	Mobile   string    `gorm:"size:20;comment:联系方式" json:"mobile"`
	Status   int8      `gorm:"default:1;comment:用户状态((1:正常;0:禁用))" json:"status"`
	Email    string    `gorm:"size:128;comment:用户邮箱" json:"email"`
	Deleted  int8      `gorm:"default:0;comment:逻辑删除标识(0:未删除;1:已删除)" json:"deleted"`
	Roles    []SysRole `gorm:"many2many:sys_user_role;joinForeignKey:user_id;joinReferences:role_id"`
}

type UserAuthInfo struct {
	UserId    int64    `json:"userId"`
	Username  string   `json:"username"`
	Nickname  string   `json:"nickname"`
	DeptId    int64    `json:"deptId"`
	Password  string   `json:"password"`
	Status    int      `json:"status"`
	Roles     []string `json:"roles"`
	Perms     []string `json:"perms"`
	DataScope int      `json:"dataScope"`
}
