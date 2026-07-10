package read

// MenuOptionRead 菜单选项读模型（用于下拉选择）
type MenuOptionRead struct {
	// 菜单ID
	ID int64 `json:"id"`
	// 父菜单ID
	ParentID int64 `json:"parentId"`
	// 菜单名称
	Name string `json:"name"`
	// 菜单类型(1-菜单；2-目录；3-外链；4-按钮权限)
	Type int8 `json:"type"`
	// 排序
	Sort int `json:"sort"`
	// 子菜单选项（非数据库字段，由业务逻辑组装）
	Children []MenuOptionRead `gorm:"-" json:"children,omitempty"`
}
