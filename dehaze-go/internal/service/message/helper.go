package message

import (
	"encoding/json"
	"time"
)

const timeFormat = "2006-01-02 15:04:05"

var messageTypeLabels = map[string]string{
	"inbox":          "站内信",
	"announcement":   "系统公告",
	"business":       "业务通知",
	"member":         "会员通知",
	"alert":          "告警通知",
	"critical_alert": "严重告警",
}

var announcementTypeLabels = map[string]string{
	"maintenance": "系统维护",
	"feature":     "功能更新",
	"activity":    "活动通知",
	"operation":   "运营公告",
}

var targetScopeLabels = map[string]string{
	"all":        "全体用户",
	"level":      "按会员等级",
	"specified":  "指定用户",
}

var announcementStatusLabels = map[int]string{
	1: "草稿",
	2: "待发送",
	3: "已发送",
	4: "已取消",
}

var importanceLabels = map[int]string{
	1: "普通",
	2: "重要",
}

func formatTime(t *time.Time) string {
	if t == nil {
		return ""
	}
	return t.Format(timeFormat)
}

func formatTimeVal(t time.Time) string {
	return t.Format(timeFormat)
}

func formatTimeStr(t *time.Time) string {
	if t == nil {
		return ""
	}
	return t.Format("15:04:05")
}

func toJSONString(v interface{}) string {
	if v == nil {
		return ""
	}
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(b)
}

func parseJSONToInterface(s string) interface{} {
	if s == "" || s == "null" {
		return nil
	}
	var v interface{}
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return nil
	}
	return v
}

func summary(content string) string {
	if len([]rune(content)) <= 50 {
		return content
	}
	runes := []rune(content)
	return string(runes[:50])
}
