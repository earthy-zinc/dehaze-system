package read

// Option 下拉选项读模型
type Option struct {
	Value    int64    `json:"value"`
	Label    string   `json:"label"`
	Code     string   `json:"code,omitempty" gorm:"column:code"`
	Children []Option `json:"children,omitempty" gorm:"-"`
}
