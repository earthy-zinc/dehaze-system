package read

// RoleForm 角色表单读模型
type RoleForm struct {
	ID        *int64 `json:"id"`
	Name      string `json:"name"`
	Code      string `json:"code"`
	Sort      int    `json:"sort"`
	Status    int8   `json:"status"`
	DataScope int8   `json:"dataScope"`
}
