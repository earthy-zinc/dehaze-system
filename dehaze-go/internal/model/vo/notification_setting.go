package vo

// NotificationSettingsVO 通知偏好设置
type NotificationSettingsVO struct {
	PushEnabled bool                   `json:"pushEnabled"`
	DndEnabled  bool                   `json:"dndEnabled"`
	DndStart    string                 `json:"dndStart"`
	DndEnd      string                 `json:"dndEnd"`
	Preferences NotificationPreferences `json:"preferences"`
}

// NotificationPreferences 细粒度偏好
type NotificationPreferences struct {
	TypeChannels   map[string]TypeChannel `json:"typeChannels"`
	ModuleSwitches map[string]bool        `json:"moduleSwitches"`
}

// TypeChannel 类型渠道开关
type TypeChannel struct {
	Push bool `json:"push"`
}
