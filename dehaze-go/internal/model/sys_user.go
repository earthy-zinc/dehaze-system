package model

// SysUser 用户信息表
type SysUser struct {
	BaseModel
	Username string    `gorm:"column:username;type:varchar(64);uniqueIndex:idx_sys_user_username;comment:用户名" json:"username"`
	Nickname string    `gorm:"column:nickname;type:varchar(64);comment:昵称" json:"nickname"`
	Gender   int8      `gorm:"column:gender;type:tinyint;default:1;comment:性别((1:男;2:女))" json:"gender"`
	Password string    `gorm:"column:password;type:varchar(100);comment:密码" json:"password"`
	DeptID   int64     `gorm:"column:dept_id;type:bigint;comment:部门ID" json:"deptId"`
	Avatar   string    `gorm:"column:avatar;type:text;comment:用户头像" json:"avatar"`
	Mobile   string    `gorm:"column:mobile;type:varchar(20);comment:联系方式" json:"mobile"`
	Status   int8      `gorm:"column:status;type:tinyint;default:1;comment:用户状态((1:正常;0:禁用))" json:"status"`
	Email    string    `gorm:"column:email;type:varchar(128);comment:用户邮箱" json:"email"`
	Deleted  int8      `gorm:"column:deleted;type:tinyint;default:0;comment:逻辑删除标识(0:未删除;1:已删除)" json:"deleted"`
	Roles    []SysRole `gorm:"many2many:sys_user_role;joinForeignKey:user_id;joinReferences:role_id"`
	CreateBy int64     `gorm:"column:create_by;type:bigint;comment:创建人ID" json:"createBy"`
	UpdateBy int64     `gorm:"column:update_by;type:bigint;comment:修改人ID" json:"updateBy"`
}

type UserAuthInfo struct {
	UserId    int64    `json:"userId"`
	Username  string   `json:"username"`
	Nickname  string   `json:"nickname"`
	DeptId    int64    `json:"deptId"` // 修复类型为int64以匹配Java的Long类型
	Password  string   `json:"password"`
	Status    int8     `json:"status"`
	Roles     []string `gorm:"-" json:"roles"`
	Perms     []string `gorm:"-" json:"perms"`
	DataScope int8     `gorm:"-" json:"dataScope"`
}
