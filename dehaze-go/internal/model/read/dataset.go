package read

import "time"

// Dataset 数据集读模型
type Dataset struct {
	ID          int64     `json:"id"`
	ParentID    int64     `json:"parentId"`
	Type        string    `json:"type"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Path        string    `json:"path"`
	Size        string    `json:"size"`
	Children    []Dataset `json:"children"`
	CreateTime  time.Time `json:"createTime"`
	UpdateTime  time.Time `json:"updateTime"`
	Status      int       `json:"status"`
}
