package read

import "time"

// DatasetForm 数据集表单读模型
type DatasetForm struct {
	ID          *int64     `json:"id"`
	ParentID    int64      `json:"parentId"`
	Type        string     `json:"type"`
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Path        string     `json:"path"`
	Status      int8       `json:"status"`
	CreateTime  time.Time  `json:"createTime"`
	UpdateTime  time.Time  `json:"updateTime"`
}
