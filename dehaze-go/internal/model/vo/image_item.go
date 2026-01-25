package vo

// ImageItemVO 图片项视图对象
type ImageItemVO struct {
	// 数据项ID
	ID int64 `json:"id"`
	// 数据集ID
	DatasetID int64 `json:"datasetId"`
	// 数据项名称
	Name string `json:"name"`
	// 图片数量
	ImageCount int `json:"imageCount"`
	// 模糊图片
	HazyImages []ImageUrlVO `json:"hazyImages"`
	// 创建时间
	CreateTime string `json:"createTime"`
	// 更新时间
	UpdateTime string `json:"updateTime"`
}
