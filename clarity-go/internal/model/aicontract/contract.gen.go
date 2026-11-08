// GENERATED FROM config/schema/ai-conversation.schema.json — DO NOT EDIT.
// 由 scripts/tooling/gen-ai-contract.mjs 生成；改契约请改 Schema 后重新运行生成器。
// 枚举字段类型为 string，取值见 const 块与字段注释；时间字段为 string（ISO-8601）。

package aicontract

// Role 取值: system | user | assistant | tool
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// MessageStatus 取值: running | completed | failed | canceled | awaiting_input
const (
	MessageStatusRunning       = "running"
	MessageStatusCompleted     = "completed"
	MessageStatusFailed        = "failed"
	MessageStatusCanceled      = "canceled"
	MessageStatusAwaitingInput = "awaiting_input"
)

// BlockKind 取值: text | thinking | tool_use
const (
	BlockKindText     = "text"
	BlockKindThinking = "thinking"
	BlockKindToolUse  = "tool_use"
)

// ToolStatus 取值: running | success | failed | skipped
const (
	ToolStatusRunning = "running"
	ToolStatusSuccess = "success"
	ToolStatusFailed  = "failed"
	ToolStatusSkipped = "skipped"
)

// PlanStatus 取值: pending | running | completed | failed
const (
	PlanStatusPending   = "pending"
	PlanStatusRunning   = "running"
	PlanStatusCompleted = "completed"
	PlanStatusFailed    = "failed"
)

// InterruptType 取值: confirm | quota | async_wait | plan_approve
const (
	InterruptTypeConfirm     = "confirm"
	InterruptTypeQuota       = "quota"
	InterruptTypeAsyncWait   = "async_wait"
	InterruptTypePlanApprove = "plan_approve"
)

// ConfirmKind 取值: algorithm_recommend | tool_permission | dangerous_op | write_conflict
const (
	ConfirmKindAlgorithmRecommend = "algorithm_recommend"
	ConfirmKindToolPermission     = "tool_permission"
	ConfirmKindDangerousOp        = "dangerous_op"
	ConfirmKindWriteConflict      = "write_conflict"
)

type ToolCall struct {
	CallID  string         `json:"callId"`
	Name    string         `json:"name"`
	Input   map[string]any `json:"input"`
	Output  *string        `json:"output"`
	Summary *string        `json:"summary"`
	// 取值: running | success | failed | skipped
	Status    string  `json:"status"`
	Error     *string `json:"error"`
	LatencyMs *int64  `json:"latencyMs"`
	AgentCode *string `json:"agentCode"`
	SubAgent  bool    `json:"subAgent"`
}

type Block struct {
	ID string `json:"id"`
	// 取值: text | thinking | tool_use
	Kind       string    `json:"kind"`
	Seq        int64     `json:"seq"`
	Text       *string   `json:"text"`
	DurationMs *int64    `json:"durationMs"`
	Tool       *ToolCall `json:"tool"`
}

type PlanTask struct {
	ID          string   `json:"id"`
	Description *string  `json:"description"`
	DependsOn   []string `json:"dependsOn"`
	// 取值: pending | running | completed | failed
	Status   *string `json:"status"`
	Paradigm *string `json:"paradigm"`
}

type PlanRevision struct {
	RevisionNo int64   `json:"revisionNo"`
	Reason     string  `json:"reason"`
	Actor      *string `json:"actor"`
	At         *string `json:"at"`
}

type Plan struct {
	Phase *string `json:"phase"`
	// 取值: pending | running | completed | failed
	Status           string         `json:"status"`
	Tasks            []PlanTask     `json:"tasks"`
	Revisions        []PlanRevision `json:"revisions"`
	AwaitingApproval bool           `json:"awaitingApproval"`
}

type Recommendation struct {
	ModelID   string  `json:"modelId"`
	ModelName string  `json:"modelName"`
	Reason    *string `json:"reason"`
	// 允许附加字段（Schema additionalProperties=true），解码时按需自行保留
}

type Alternative struct {
	ModelID    string   `json:"modelId"`
	ModelName  string   `json:"modelName"`
	MatchScore *float64 `json:"matchScore"`
	Reason     *string  `json:"reason"`
	// 允许附加字段（Schema additionalProperties=true），解码时按需自行保留
}

type Interrupt struct {
	ID string `json:"id"`
	// 取值: confirm | quota | async_wait | plan_approve
	Type string `json:"type"`
	// 取值: algorithm_recommend | tool_permission | dangerous_op | write_conflict
	ConfirmKind    *string         `json:"confirmKind"`
	Action         *string         `json:"action"`
	Recommendation *Recommendation `json:"recommendation"`
	Alternatives   []Alternative   `json:"alternatives"`
	UpgradeTip     *string         `json:"upgradeTip"`
	UsedDaily      *int64          `json:"usedDaily"`
	DailyLimit     *int64          `json:"dailyLimit"`
	UsedMonthly    *int64          `json:"usedMonthly"`
	MonthlyLimit   *int64          `json:"monthlyLimit"`
	EstDuration    *string         `json:"estDuration"`
	ImageCount     *int64          `json:"imageCount"`
	Plan           []PlanTask      `json:"plan"`
	Reason         *string         `json:"reason"`
	Detail         *string         `json:"detail"`
	CreatedAt      string          `json:"createdAt"`
	ExpiresAt      string          `json:"expiresAt"`
	Actionable     bool            `json:"actionable"`
}

type SubAgentUsage struct {
	AgentCode         string `json:"agentCode"`
	InputTokens       int64  `json:"inputTokens"`
	OutputTokens      int64  `json:"outputTokens"`
	CachedInputTokens int64  `json:"cachedInputTokens"`
	Credits           int64  `json:"credits"`
}

type Usage struct {
	InputTokens       int64           `json:"inputTokens"`
	OutputTokens      int64           `json:"outputTokens"`
	CachedInputTokens int64           `json:"cachedInputTokens"`
	Credits           int64           `json:"credits"`
	SubAgents         []SubAgentUsage `json:"subAgents"`
}

type Message struct {
	ID              int64  `json:"id"`
	ConversationID  int64  `json:"conversationId"`
	ParentMessageID *int64 `json:"parentMessageId"`
	// 取值: system | user | assistant | tool
	Role string `json:"role"`
	// 取值: running | completed | failed | canceled | awaiting_input
	Status           string     `json:"status"`
	Model            *string    `json:"model"`
	Error            *string    `json:"error"`
	Blocks           []Block    `json:"blocks"`
	Plan             *Plan      `json:"plan"`
	PendingInterrupt *Interrupt `json:"pendingInterrupt"`
	Usage            Usage      `json:"usage"`
	ArtifactIDs      []int64    `json:"artifactIds"`
	MemoryIDs        []int64    `json:"memoryIds"`
	Suggestions      []string   `json:"suggestions"`
	Edited           bool       `json:"edited"`
	SiblingIndex     *int64     `json:"siblingIndex"`
	SiblingCount     *int64     `json:"siblingCount"`
	CreateTime       string     `json:"createTime"`
	UpdateTime       *string    `json:"updateTime"`
}

type MessageList struct {
	List    []Message `json:"list"`
	Total   int64     `json:"total"`
	HasMore bool      `json:"hasMore"`
}

type MessageSend struct {
	Content string  `json:"content"`
	Model   *string `json:"model"`
}

type MessageResume struct {
	Confirm  *bool          `json:"confirm"`
	Params   map[string]any `json:"params"`
	PlanEdit map[string]any `json:"planEdit"`
}

type Artifact struct {
	ID             int64          `json:"id"`
	ConversationID int64          `json:"conversationId"`
	MessageID      int64          `json:"messageId"`
	Type           string         `json:"type"`
	RefType        *string        `json:"refType"`
	RefID          *int64         `json:"refId"`
	Summary        map[string]any `json:"summary"`
	IsInvalid      bool           `json:"isInvalid"`
	CreateTime     *string        `json:"createTime"`
}

type Feedback struct {
	ID             *int64 `json:"id"`
	MessageID      int64  `json:"messageId"`
	UserID         int64  `json:"userId"`
	ConversationID *int64 `json:"conversationId"`
	// 取值: 1 | -1
	Rating     int64    `json:"rating"`
	Tags       []string `json:"tags"`
	Comment    *string  `json:"comment"`
	CreateTime *string  `json:"createTime"`
	UpdateTime *string  `json:"updateTime"`
}

type MessageStartEvent struct {
	StreamSessionID  string  `json:"streamSessionId"`
	UserMessage      Message `json:"userMessage"`
	AssistantMessage Message `json:"assistantMessage"`
}

type BlockStartEvent struct {
	MessageID int64 `json:"messageId"`
	Block     Block `json:"block"`
}

type BlockDeltaEvent struct {
	MessageID int64  `json:"messageId"`
	BlockID   string `json:"blockId"`
	Delta     string `json:"delta"`
}

type BlockStopEvent struct {
	MessageID int64 `json:"messageId"`
	Block     Block `json:"block"`
}

type PlanUpdateEvent struct {
	MessageID int64 `json:"messageId"`
	Plan      Plan  `json:"plan"`
}

type InterruptEvent struct {
	MessageID int64     `json:"messageId"`
	Interrupt Interrupt `json:"interrupt"`
}

type UsageEvent struct {
	MessageID int64 `json:"messageId"`
	Usage     Usage `json:"usage"`
}

type SuggestionsEvent struct {
	MessageID   int64    `json:"messageId"`
	Suggestions []string `json:"suggestions"`
}

type MessageEndEvent struct {
	MessageID int64 `json:"messageId"`
	// 取值: running | completed | failed | canceled | awaiting_input
	Status string  `json:"status"`
	Error  *string `json:"error"`
	Usage  *Usage  `json:"usage"`
}

type ErrorEvent struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	MessageID *int64 `json:"messageId"`
}
