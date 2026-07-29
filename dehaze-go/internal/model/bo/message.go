package bo

import "encoding/json"

// MessageSendForm 内部消息发送表单
type MessageSendForm struct {
	TemplateCode string                 `json:"templateCode"`
	Type         string                 `json:"type"`
	Title        string                 `json:"title"`
	Content      string                 `json:"content"`
	RecipientIDs []int64                `json:"recipientIds"`
	BizModule    string                 `json:"bizModule"`
	BizID        string                 `json:"bizId"`
	Priority     int                    `json:"priority"`
	JumpURL      string                 `json:"jumpUrl"`
	Variables    map[string]string      `json:"variables"`
	Extra        map[string]interface{} `json:"extra"`
}

// AnnouncementForm 公告表单
type AnnouncementForm struct {
	Title        string                 `json:"title"`
	Content      string                 `json:"content"`
	Type         string                 `json:"type"`
	Importance   int                    `json:"importance"`
	TargetScope  string                 `json:"targetScope"`
	TargetParams map[string]interface{} `json:"targetParams"`
	SendTime     *string                `json:"sendTime"`
	ExpireTime   *string                `json:"expireTime"`
}

// MessageTemplateForm 模板编辑表单
type MessageTemplateForm struct {
	Name            *string                `json:"name"`
	TitleTemplate   *string                `json:"titleTemplate"`
	ContentTemplate *string                `json:"contentTemplate"`
	Priority        *int                   `json:"priority"`
	Channels        map[string]bool        `json:"channels"`
	Status          *int                   `json:"status"`
}

// NotificationSettingForm 通知设置表单
type NotificationSettingForm struct {
	PushEnabled *bool                  `json:"pushEnabled"`
	DndEnabled  *bool                  `json:"dndEnabled"`
	DndStart    *string                `json:"dndStart"`
	DndEnd      *string                `json:"dndEnd"`
	Preferences *NotificationPrefsForm `json:"preferences"`
}

// NotificationPrefsForm 通知偏好子表单
type NotificationPrefsForm struct {
	TypeChannels   map[string]TypeChannelForm `json:"typeChannels"`
	ModuleSwitches map[string]bool            `json:"moduleSwitches"`
}

// TypeChannelForm 类型渠道开关表单
type TypeChannelForm struct {
	Push bool `json:"push"`
}

// ToJSONString 将map转为JSON字符串
func ToJSONString(v any) string {
	if v == nil {
		return ""
	}
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(b)
}
