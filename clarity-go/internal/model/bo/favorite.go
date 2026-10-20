package bo

// FavoriteForm 收藏表单
type FavoriteForm struct {
	TargetType string `json:"targetType" binding:"required"`
	TargetID   int64  `json:"targetId" binding:"required"`
}
