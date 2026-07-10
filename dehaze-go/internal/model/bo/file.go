package bo

// FileBO 文件业务对象
type FileBO struct {
	// 文件名
	Name string `json:"name"`
	// 文件对象名
	ObjectName string `json:"objectName"`
	// 文件扩展名
	Extension string `json:"extension"`
	// 文件MD5值
	MD5 string `json:"md5"`
	// 文件路径
	Path string `json:"path"`
	// 文件大小
	Size int64 `json:"size"`
	// 文件URL
	URL string `json:"url"`
}

// ItemFileUpdateForm 图片信息更新表单
type ItemFileUpdateForm struct {
	// 图片类型（clear/hazy/depth/segment）
	Type *string `json:"type"`
	// 场景类型
	SceneType *string `json:"sceneType"`
	// 雾霾程度（light/medium/heavy）
	HazeLevel *string `json:"hazeLevel"`
	// 描述
	Description *string `json:"description"`
}
