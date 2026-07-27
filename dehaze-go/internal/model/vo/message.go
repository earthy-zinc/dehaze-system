package vo

// MessageVO 消息列表项
type MessageVO struct {
	ID            int64  `json:"id"`
	Type          string `json:"type"`
	TypeLabel     string `json:"typeLabel"`
	Title         string `json:"title"`
	Summary       string `json:"summary"`
	Priority      int    `json:"priority"`
	ReadStatus    int    `json:"readStatus"`
	SenderType    int    `json:"senderType"`
	JumpURL       string `json:"jumpUrl"`
	CreateTime    string `json:"createTime"`
}

// MessageDetailVO 消息详情
type MessageDetailVO struct {
	ID             int64       `json:"id"`
	Type           string      `json:"type"`
	TypeLabel      string      `json:"typeLabel"`
	Title          string      `json:"title"`
	Content        string      `json:"content"`
	Priority       int         `json:"priority"`
	SenderType     int         `json:"senderType"`
	SenderTypeLabel string     `json:"senderTypeLabel"`
	ReadStatus     int         `json:"readStatus"`
	ReadTime       string      `json:"readTime"`
	JumpURL        string      `json:"jumpUrl"`
	Extra          interface{} `json:"extra"`
	CreateTime     string      `json:"createTime"`
}

// UnreadCountVO 未读消息数
type UnreadCountVO struct {
	Count int64 `json:"count"`
}

// ReadAllResultVO 全部已读结果
type ReadAllResultVO struct {
	AffectedCount int64 `json:"affectedCount"`
}

// MessageSendResultVO 发送消息结果
type MessageSendResultVO struct {
	MessageIDs []int64 `json:"messageIds"`
}
