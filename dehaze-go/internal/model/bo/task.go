package bo

// ExportOptions 导出选项配置
type ExportOptions struct {
	// 文件组织方式：by_item（按数据项）, flat（扁平结构）
	Structure string `json:"structure"`
	// 包含的类型：clear（清晰图）, hazy（有雾图）等
	IncludeTypes []string `json:"includeTypes"`
	// 是否包含缩略图
	IncludeThumbnail bool `json:"includeThumbnail"`
}

// ExportTaskCreateForm 导出任务创建表单
type ExportTaskCreateForm struct {
	// 导出类型：dataset, dataset_item, batch_items, custom
	Type string `json:"type" validate:"required"`
	// 目标ID（导出单个资源时使用）
	TargetID int64 `json:"targetId"`
	// 目标ID列表（批量导出时使用）
	TargetIDs []int64 `json:"targetIds"`
	// 导出选项配置
	Options ExportOptions `json:"options"`
}

// BatchDeleteBatchForm 批量删除表单
type BatchDeleteForm struct {
	// 要删除的ID列表
	IDs []int64 `json:"ids" validate:"required,min=1"`
	// 是否强制删除（跳过检查）
	Force bool `json:"force"`
}

// BatchDownloadForm 批量下载表单
type BatchDownloadForm struct {
	// 数据集项ID列表
	ItemID int64 `json:"itemId" validate:"required"`
	// 文件类型过滤器
	Types []string `json:"types"`
	// 是否包含缩略图
	IncludeThumbnail bool `json:"includeThumbnail"`
}
