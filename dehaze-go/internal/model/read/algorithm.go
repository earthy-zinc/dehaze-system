package read

// Algorithm 算法读模型
type Algorithm struct {
	ID          int64       `json:"id"`
	ParentID    int64       `json:"parentId"`
	Name        string      `json:"name"`
	Type        string      `json:"type"`
	Img         string      `json:"img"`
	Description string      `json:"description"`
	Path        string      `json:"path"`
	Flops       string      `json:"flops"`
	Params      string      `json:"params"`
	ImportPath  string      `json:"importPath"`
	Status      int         `json:"status"`
	Size        string      `json:"size"`
	Children    []Algorithm `json:"children"`
}

func (a *Algorithm) GetID() int64 {
	return a.ID
}

func (a *Algorithm) GetParentID() int64 {
	return a.ParentID
}
