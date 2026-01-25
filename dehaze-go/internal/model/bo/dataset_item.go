package bo

// DatasetItemBO 数据集项业务对象
type DatasetItemBO struct {
	FileBO
	Type        string `json:"type"`
	Description string `json:"description"`
}
