package vo

import "time"

type DatasetStatistics struct {
	ItemCount          int64            `json:"itemCount"`
	FileCount          int64            `json:"fileCount"`
	TotalSize          int64            `json:"totalSize"`
	ClearCount         int64            `json:"clearCount"`
	HazyCount          int64            `json:"hazyCount"`
	SceneDistribution  map[string]int64 `json:"sceneDistribution"`
	HazeDistribution   map[string]int64 `json:"hazeDistribution"`
	FormatDistribution map[string]int64 `json:"formatDistribution"`
}

type DatasetVO struct {
	ID          int64              `json:"id"`
	ParentID    int64              `json:"parentId"`
	Type        string             `json:"type"`
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Path        string             `json:"path"`
	Size        string             `json:"size"`
	HasChildren bool               `json:"hasChildren"`
	Children    []DatasetVO        `json:"children"`
	Status      int                `json:"status"`
	Statistics  *DatasetStatistics `json:"statistics"`
	Total       int64              `json:"total"`
	CreateTime  time.Time          `json:"createTime"`
	UpdateTime  time.Time          `json:"updateTime"`
}
