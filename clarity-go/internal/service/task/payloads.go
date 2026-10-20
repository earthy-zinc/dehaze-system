package task

// ThumbnailTaskPayload 单文件缩略图任务负载
// 用于图片缩略图生成
// itemID 为数据项 ID
// fileID 为源文件 ID
// itemFileID 为项文件关联 ID
type ThumbnailTaskPayload struct {
	ItemID     int64 `json:"itemId"`
	FileID     int64 `json:"fileId"`
	ItemFileID int64 `json:"itemFileId"`
}

// ThumbnailBatchPayload 批量缩略图任务负载
// 用于数据集批量缩略图生成
type ThumbnailBatchPayload struct {
	DatasetID int64   `json:"datasetId"`
	ItemID    int64   `json:"itemId"`
	FileIDs   []int64 `json:"fileIds"`
}

// FileDeletionPayload 文件删除任务负载
// 用于批量删除物理文件
type FileDeletionPayload struct {
	DatasetID int64    `json:"datasetId"`
	FilePaths []string `json:"filePaths"`
}
