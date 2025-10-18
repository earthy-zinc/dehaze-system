package model

// SysDept 部门表
type SysDept struct {
	BaseModel
	Name     string `gorm:"column:name;type:varchar(64);not null;default:'';comment:部门名称" json:"name"`
	ParentID int64  `gorm:"column:parent_id;type:bigint;not null;default:0;comment:父节点id" json:"parentId"`
	TreePath string `gorm:"column:tree_path;type:varchar(255);default:'';comment:父节点id路径" json:"treePath"`
	Sort     int    `gorm:"column:sort;type:int;default:0;comment:显示顺序" json:"sort"`
	Status   int8   `gorm:"column:status;type:tinyint;not null;default:1;comment:状态(1:正常;0:禁用)" json:"status"`
	Deleted  int8   `gorm:"column:deleted;type:tinyint;default:0;comment:逻辑删除标识(1:已删除;0:未删除)" json:"deleted"`
	CreateBy int64  `gorm:"column:create_by;type:bigint;comment:创建人ID" json:"createBy"`
	UpdateBy int64  `gorm:"column:update_by;type:bigint;comment:修改人ID" json:"updateBy"`
}
