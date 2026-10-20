package vo

// Option 下拉选项对象
type Option struct {
	// 选项的值
	Value interface{} `json:"value"`
	// 选项的标签
	Label string `json:"label"`
	// 子选项列表
	Children []Option `json:"children,omitempty"`
}
