package model

import "time"

// SysAiConversation AI 对话会话
type SysAiConversation struct {
	ID                     int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	UserID                 int64      `gorm:"column:user_id" json:"userId"`
	Title                  string     `gorm:"column:title" json:"title"`
	Model                  *string    `gorm:"column:model" json:"model"`
	AgentCode              *string    `gorm:"column:agent_code" json:"agentCode"`
	AgentVersion           *int       `gorm:"column:agent_version" json:"agentVersion"`
	Summary                *string    `gorm:"column:summary" json:"summary"`
	SummaryUptoMessageID   *int64     `gorm:"column:summary_upto_message_id" json:"summaryUptoMessageId"`
	SystemPrompt           *string    `gorm:"column:system_prompt" json:"systemPrompt"`
	ModelConfig            string     `gorm:"column:model_config;type:json;default:null" json:"modelConfig"`
	SuggestionsEnabled     int        `gorm:"column:suggestions_enabled" json:"suggestionsEnabled"`
	APIKeyID               *int64     `gorm:"column:api_key_id" json:"apiKeyId"`
	MessageCount           int        `gorm:"column:message_count" json:"messageCount"`
	LastMessageAt          *time.Time `gorm:"column:last_message_at" json:"lastMessageAt"`
	CurrentBranchMessageID *int64     `gorm:"column:current_branch_message_id" json:"currentBranchMessageId"`
	LastReadMessageID      *int64     `gorm:"column:last_read_message_id" json:"lastReadMessageId"`
	Pinned                 int        `gorm:"column:pinned" json:"pinned"`
	PinnedAt               *time.Time `gorm:"column:pinned_at" json:"pinnedAt"`
	DeleteTime             *time.Time `gorm:"column:delete_time" json:"deleteTime"`
	TitleSource            string     `gorm:"column:title_source" json:"titleSource"`
	Status                 int        `gorm:"column:status" json:"status"`
	Deleted                int64      `gorm:"column:deleted" json:"deleted"`
	CreateBy               *int64     `gorm:"column:create_by" json:"createBy"`
	UpdateBy               *int64     `gorm:"column:update_by" json:"updateBy"`
	CreateTime             time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime             *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiConversation) TableName() string { return "sys_ai_conversation" }

// SysAiMessage AI 对话消息
type SysAiMessage struct {
	ID                int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	ConversationID    int64      `gorm:"column:conversation_id" json:"conversationId"`
	ParentMessageID   *int64     `gorm:"column:parent_message_id" json:"parentMessageId"`
	Role              string     `gorm:"column:role" json:"role"`
	Content           *string    `gorm:"column:content" json:"content"`
	ToolCalls         string     `gorm:"column:tool_calls;type:json;default:null" json:"toolCalls"`
	ToolCallID        *string    `gorm:"column:tool_call_id" json:"toolCallId"`
	Model             *string    `gorm:"column:model" json:"model"`
	Status            int        `gorm:"column:status" json:"status"`
	Error             *string    `gorm:"column:error" json:"error"`
	Metadata          string     `gorm:"column:metadata;type:json;default:null" json:"metadata"`
	InputTokens       int        `gorm:"column:input_tokens" json:"inputTokens"`
	OutputTokens      int        `gorm:"column:output_tokens" json:"outputTokens"`
	CachedInputTokens int        `gorm:"column:cached_input_tokens" json:"cachedInputTokens"`
	Credits           int64      `gorm:"column:credits" json:"credits"`
	TaskID            *string    `gorm:"column:task_id" json:"taskId"`
	UsedMemoryIDs     string     `gorm:"column:used_memory_ids;type:json;default:null" json:"usedMemoryIds"`
	Edited            int        `gorm:"column:edited" json:"edited"`
	OriginalContent   *string    `gorm:"column:original_content" json:"originalContent"`
	Deleted           int64      `gorm:"column:deleted" json:"deleted"`
	CreateBy          *int64     `gorm:"column:create_by" json:"createBy"`
	UpdateBy          *int64     `gorm:"column:update_by" json:"updateBy"`
	CreateTime        time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime        *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiMessage) TableName() string { return "sys_ai_message" }

// SysAiAgentThought 推理步骤
type SysAiAgentThought struct {
	ID             int64     `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	MessageID      int64     `gorm:"column:message_id" json:"messageId"`
	ConversationID int64     `gorm:"column:conversation_id" json:"conversationId"`
	Position       int       `gorm:"column:position" json:"position"`
	AgentCode      *string   `gorm:"column:agent_code" json:"agentCode"`
	IsSubagent     int       `gorm:"column:is_subagent" json:"isSubagent"`
	Thought        *string   `gorm:"column:thought" json:"thought"`
	Tool           *string   `gorm:"column:tool" json:"tool"`
	ToolInput      string    `gorm:"column:tool_input;type:json;default:null" json:"toolInput"`
	Observation    *string   `gorm:"column:observation" json:"observation"`
	Summary        *string   `gorm:"column:summary" json:"summary"`
	Status         int       `gorm:"column:status" json:"status"`
	LatencyMs      *int      `gorm:"column:latency_ms" json:"latencyMs"`
	Error          *string   `gorm:"column:error" json:"error"`
	CreateTime     time.Time `gorm:"column:create_time;autoCreateTime" json:"createTime"`
}

func (SysAiAgentThought) TableName() string { return "sys_ai_agent_thought" }

// SysAiMessageFeedback 消息反馈
type SysAiMessageFeedback struct {
	ID             int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	MessageID      int64      `gorm:"column:message_id" json:"messageId"`
	UserID         int64      `gorm:"column:user_id" json:"userId"`
	ConversationID *int64     `gorm:"column:conversation_id" json:"conversationId"`
	Model          *string    `gorm:"column:model" json:"model"`
	Source         string     `gorm:"column:source" json:"source"`
	Rating         int        `gorm:"column:rating" json:"rating"`
	Tags           string     `gorm:"column:tags;type:json;default:null" json:"tags"`
	Comment        *string    `gorm:"column:comment" json:"comment"`
	Processed      int        `gorm:"column:processed" json:"processed"`
	ProcessTime    *time.Time `gorm:"column:process_time" json:"processTime"`
	Deleted        int64      `gorm:"column:deleted" json:"deleted"`
	CreateBy       *int64     `gorm:"column:create_by" json:"createBy"`
	UpdateBy       *int64     `gorm:"column:update_by" json:"updateBy"`
	CreateTime     time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime     *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiMessageFeedback) TableName() string { return "sys_ai_message_feedback" }

// SysAiArtifact AI 中间产物（只追加，无逻辑删除）
type SysAiArtifact struct {
	ID             int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	ConversationID int64      `gorm:"column:conversation_id" json:"conversationId"`
	MessageID      int64      `gorm:"column:message_id" json:"messageId"`
	Type           string     `gorm:"column:type" json:"type"`
	RefType        *string    `gorm:"column:ref_type" json:"refType"`
	RefID          *int64     `gorm:"column:ref_id" json:"refId"`
	Summary        string     `gorm:"column:summary;type:json;default:null" json:"summary"`
	IsInvalid      int        `gorm:"column:is_invalid" json:"isInvalid"`
	CreateBy       *int64     `gorm:"column:create_by" json:"createBy"`
	UpdateBy       *int64     `gorm:"column:update_by" json:"updateBy"`
	CreateTime     time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime     *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiArtifact) TableName() string { return "sys_ai_artifact" }
