package dto

// ChatMessage WebSocket 消息体
type ChatMessage struct {
	// 发送者
	Sender string `json:"sender"`
	// 消息内容
	Content string `json:"content"`
}