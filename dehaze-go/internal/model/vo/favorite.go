package vo

// FavoriteVO 收藏列表视图对象
type FavoriteVO struct {
	ID         int64  `json:"id"`
	UserID     int64  `json:"userId"`
	TargetType string `json:"targetType"`
	TargetID   int64  `json:"targetId"`
	TargetName string `json:"targetName"`
	IsInvalid  bool   `json:"isInvalid"`
	CreateTime string `json:"createTime"`
}

// FavoriteStatusVO 收藏状态视图对象
type FavoriteStatusVO struct {
	TargetType string `json:"targetType"`
	TargetID   int64  `json:"targetId"`
	Favorited  bool   `json:"favorited"`
}

// FavoriteCountVO 收藏数量视图对象（按类型分组）
type FavoriteCountVO struct {
	TargetType string `json:"targetType"`
	Count      int64  `json:"count"`
}
