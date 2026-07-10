package read

// Option 下拉选项读模型
type Option struct {
	Value    any `json:"value"`
	Label    string      `json:"label"`
	Children []Option    `json:"children,omitempty"`
}
