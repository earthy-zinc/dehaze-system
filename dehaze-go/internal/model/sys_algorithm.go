package model

import "time"

// SysAlgorithm 算法模型表
type SysAlgorithm struct {
	BaseModel
	ParentID    int64      `gorm:"column:parent_id;type:bigint;default:0;comment:模型的父id" json:"parentId"`
	Type        string     `gorm:"column:type;type:varchar(100);default:'';comment:模型类型" json:"type"`
	Version     *string    `gorm:"column:version;type:varchar(50);comment:算法版本号" json:"version"`
	Name        string     `gorm:"column:name;type:varchar(64);not null;comment:模型名称" json:"name"`
	Img         string     `gorm:"column:img;type:text;comment:模型图片" json:"img"`
	Path        string     `gorm:"column:path;type:varchar(255);default:'';comment:模型存储路径" json:"path"`
	Size        string     `gorm:"column:size;type:varchar(100);comment:模型大小" json:"size"`
	Params      string     `gorm:"column:params;type:varchar(255);comment:模型参数" json:"params"`
	Flops       string     `gorm:"column:flops;type:varchar(255);comment:模型浮点运算次数" json:"flops"`
	ImportPath  string     `gorm:"column:import_path;type:varchar(255);comment:模型代码导入路径" json:"importPath"`
	Description string     `gorm:"column:description;type:varchar(2048);comment:针对该模型的详细描述" json:"description"`
	Status      int8       `gorm:"column:status;type:tinyint;not null;default:1;comment:状态(1:草稿;2:测试中;3:待审核;4:已发布;5:已停用;6:已归档)" json:"status"`
	AuditBy     *int64     `gorm:"column:audit_by;type:bigint;comment:审核人ID" json:"auditBy"`
	AuditTime   *time.Time `gorm:"column:audit_time;type:datetime;comment:审核时间" json:"auditTime"`
	AuditRemark *string    `gorm:"column:audit_remark;type:varchar(500);comment:审核备注" json:"auditRemark"`
	Deleted     int8       `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识(0:未删除;1:已删除)" json:"deleted"`
}
