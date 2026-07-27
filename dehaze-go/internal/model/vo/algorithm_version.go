package vo

import "time"

// AlgorithmVersionVO 算法版本视图对象
type AlgorithmVersionVO struct {
	// 版本ID
	ID int64 `json:"id"`
	// 算法ID
	AlgorithmID int64 `json:"algorithmId"`
	// 版本号
	Version string `json:"version"`
	// 变更日志
	ChangeLog *string `json:"changeLog"`
	// 状态
	Status *int8 `json:"status"`
	// 是否当前活跃版本
	IsActive *bool `json:"isActive"`
	// 模型文件ID
	ModelFileID *int64 `json:"modelFileId"`
	// 创建时间
	CreateTime time.Time `json:"createTime"`
}
