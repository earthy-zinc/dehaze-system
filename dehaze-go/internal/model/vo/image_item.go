package vo

// ImageItemVO 图片项视图对象
type ImageItemVO struct {
	// 数据项ID
	ID int64 `json:"id"`
	// 数据集ID
	DatasetID int64 `json:"datasetId"`
	// 数据项名称
	Name string `json:"name"`
	// 场景类型
	SceneType string `json:"sceneType,omitempty"`
	// 数据项描述
	Description string `json:"description,omitempty"`
	// 图片数量
	ImageCount int `json:"imageCount"`
	// 清晰图信息
	ClearImage *ImageUrlVO `json:"clearImage,omitempty"`
	// 模糊图片
	HazyImages []ImageUrlVO `json:"hazyImages"`
	// 创建时间
	CreateTime string `json:"createTime"`
	// 更新时间
	UpdateTime string `json:"updateTime"`
}
