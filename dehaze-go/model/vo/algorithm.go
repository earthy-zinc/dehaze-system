package vo

// AlgorithmVO 算法视图对象
type AlgorithmVO struct {
	// 算法ID
	ID int64 `json:"id"`
	// 算法名称
	Name string `json:"name"`
	// 算法类型
	Type string `json:"type"`
	// 算法图片
	Img string `json:"img"`
	// 算法描述
	Description string `json:"description"`
	// 算法路径
	Path string `json:"path"`
	// 算法浮点数
	Flops string `json:"flops"`
	// 算法参数量
	Params string `json:"params"`
	// 导入路径
	ImportPath string `json:"importPath"`
	// 开启关闭状态
	Status int `json:"status"`
	// 算法大小
	Size string `json:"size"`
	// 子算法
	Children []AlgorithmVO `json:"children"`
}