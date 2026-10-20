package bo

// RolePermsBO 角色权限业务对象
type RolePermsBO struct {
	// 角色编码
	RoleCode string `json:"roleCode"`
	// 权限标识集合
	Perms []string `json:"perms"`
}
