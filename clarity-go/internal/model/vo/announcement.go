package vo

// AnnouncementVO 公告列表项
type AnnouncementVO struct {
	ID               int64  `json:"id"`
	Title            string `json:"title"`
	Type             string `json:"type"`
	TypeLabel        string `json:"typeLabel"`
	Importance       int    `json:"importance"`
	TargetScope      string `json:"targetScope"`
	TargetScopeLabel string `json:"targetScopeLabel"`
	Status           int    `json:"status"`
	StatusLabel      string `json:"statusLabel"`
	SendTime         string `json:"sendTime"`
	ExpireTime       string `json:"expireTime"`
	SentCount        int    `json:"sentCount"`
	CreateTime       string `json:"createTime"`
	CreateBy         int64  `json:"createBy"`
}

// AnnouncementDetailVO 公告详情
type AnnouncementDetailVO struct {
	ID               int64       `json:"id"`
	Title            string      `json:"title"`
	Content          string      `json:"content"`
	Type             string      `json:"type"`
	TypeLabel        string      `json:"typeLabel"`
	Importance       int         `json:"importance"`
	ImportanceLabel  string      `json:"importanceLabel"`
	TargetScope      string      `json:"targetScope"`
	TargetScopeLabel string      `json:"targetScopeLabel"`
	TargetParams     interface{} `json:"targetParams"`
	Status           int         `json:"status"`
	StatusLabel      string      `json:"statusLabel"`
	SendTime         string      `json:"sendTime"`
	ExpireTime       string      `json:"expireTime"`
	SentCount        int         `json:"sentCount"`
	CreateTime       string      `json:"createTime"`
	UpdateTime       string      `json:"updateTime"`
}

// AnnouncementSendResultVO 发送公告结果
type AnnouncementSendResultVO struct {
	SentCount int `json:"sentCount"`
}

// AnnouncementCreateResultVO 创建公告结果
type AnnouncementCreateResultVO struct {
	ID int64 `json:"id"`
}
