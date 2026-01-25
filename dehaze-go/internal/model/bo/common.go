package bo

// Id 通用ID参数
type Id struct {
	ID int64 `uri:"id" form:"id" json:"id" binding:"required"`
}
