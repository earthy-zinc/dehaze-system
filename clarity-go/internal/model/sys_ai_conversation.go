package model

import (
	"time"

	"github.com/earthyzinc/clarity-go/internal/model/aicontract"
	"gorm.io/gorm"
)

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

// SysAiMessage AI 对话消息。
//
// 消息只有一种形状：正文/思考/工具统一由 blocks(Block[]) 承载，不再有 content / tool_calls /
// tool_call_id / metadata 列；plan、pending_interrupt、suggestions 分别持久化执行计划、
// 待恢复中断与追问建议；status 是字符串枚举（running/completed/failed/canceled/awaiting_input）。
//
// blocks 为 NOT NULL 且库中无默认值：零值必须落合法 JSON 数组，故用 default:'[]' 让 GORM
// 在零值时写 "[]"（同 sys_ai_agent_version.snapshot 的处理）。status 不设 GORM 默认值：
// 状态列的非零 default 会被 GORM 在 Create 时用来吞掉显式零值，仓库守卫测试禁止。
type SysAiMessage struct {
	ID                int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	ConversationID    int64      `gorm:"column:conversation_id" json:"conversationId"`
	ParentMessageID   *int64     `gorm:"column:parent_message_id" json:"parentMessageId"`
	Role              string     `gorm:"column:role" json:"role"`
	Status            string     `gorm:"column:status" json:"status"`
	Blocks            string     `gorm:"column:blocks;type:json;not null;default:'[]'" json:"blocks"`
	Plan              string     `gorm:"column:plan;type:json;default:null" json:"plan"`
	PendingInterrupt  string     `gorm:"column:pending_interrupt;type:json;default:null" json:"pendingInterrupt"`
	Suggestions       string     `gorm:"column:suggestions;type:json;default:null" json:"suggestions"`
	Model             *string    `gorm:"column:model" json:"model"`
	Error             *string    `gorm:"column:error" json:"error"`
	InputTokens       int        `gorm:"column:input_tokens" json:"inputTokens"`
	OutputTokens      int        `gorm:"column:output_tokens" json:"outputTokens"`
	CachedInputTokens int        `gorm:"column:cached_input_tokens" json:"cachedInputTokens"`
	Credits           int64      `gorm:"column:credits" json:"credits"`
	TaskID            *string    `gorm:"column:task_id" json:"taskId"`
	UsedMemoryIDs     string     `gorm:"column:used_memory_ids;type:json;default:null" json:"usedMemoryIds"`
	SubAgentUsage     string     `gorm:"column:sub_agent_usage;type:json;default:null" json:"subAgentUsage"`
	Edited            int        `gorm:"column:edited" json:"edited"`
	Deleted           int64      `gorm:"column:deleted" json:"deleted"`
	CreateBy          *int64     `gorm:"column:create_by" json:"createBy"`
	UpdateBy          *int64     `gorm:"column:update_by" json:"updateBy"`
	CreateTime        time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime        *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

// BeforeCreate status 零值兜底：status 列禁止携带 GORM 非零 default（会被用来吞掉显式零值），
// 创建时显式补 running，避免空串落库。
func (m *SysAiMessage) BeforeCreate(*gorm.DB) error {
	if m.Status == "" {
		m.Status = aicontract.MessageStatusRunning
	}
	return nil
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
