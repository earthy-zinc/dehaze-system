package bo

// PairedImage 成对图像业务对象
type PairedImage struct {
	HazePath  []string `json:"hazePath"`
	CleanPath string   `json:"cleanPath"`
}
