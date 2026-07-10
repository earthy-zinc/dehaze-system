package read

// Algorithm 算法读模型
type Algorithm struct {
	ID          int64      `json:"id"`
	Name        string     `json:"name"`
	Type        string     `json:"type"`
	Img         string     `json:"img"`
	Description string     `json:"description"`
	Path        string     `json:"path"`
	Flops       string     `json:"flops"`
	Params      string     `json:"params"`
	ImportPath  string     `json:"importPath"`
	Status      int        `json:"status"`
	Size        string     `json:"size"`
	Children    []Algorithm `json:"children"`
}
