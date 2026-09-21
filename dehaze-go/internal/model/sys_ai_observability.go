package model

import (
	"encoding/json"
	"time"
)

// AI 可观测性（F-M08-013）实体：过程链汇总 + LLM 调用明细，均为只追加日志表，无逻辑删除。

// SysAiTrace AI 对话过程链汇总记录（每次助手回复一条）
type SysAiTrace struct {
	ID               int64           `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	TraceID          string          `gorm:"column:trace_id" json:"traceId"`
	ConversationID   int64           `gorm:"column:conversation_id" json:"conversationId"`
	MessageID        *int64          `gorm:"column:message_id" json:"messageId"`
	AgentCode        *string         `gorm:"column:agent_code" json:"agentCode"`
	TraceType        string          `gorm:"column:trace_type" json:"traceType"`
	Model            *string         `gorm:"column:model" json:"model"`
	Status           int             `gorm:"column:status" json:"status"`
	ErrorType        *string         `gorm:"column:error_type" json:"errorType"`
	DurationMs       int             `gorm:"column:duration_ms" json:"durationMs"`
	FirstTokenMs     *int            `gorm:"column:first_token_ms" json:"firstTokenMs"`
	LlmCallCount     int             `gorm:"column:llm_call_count" json:"llmCallCount"`
	TotalTokens      int             `gorm:"column:total_tokens" json:"totalTokens"`
	PromptTokens     int             `gorm:"column:prompt_tokens" json:"promptTokens"`
	CompletionTokens int             `gorm:"column:completion_tokens" json:"completionTokens"`
	CachedTokens     int             `gorm:"column:cached_tokens" json:"cachedTokens"`
	StepCount        int             `gorm:"column:step_count" json:"stepCount"`
	ContextSnapshot  json.RawMessage `gorm:"column:context_snapshot" json:"contextSnapshot"`
	ErrorDetail      json.RawMessage `gorm:"column:error_detail" json:"errorDetail"`
	CreateTime       time.Time       `gorm:"column:create_time;autoCreateTime" json:"createTime"`
}

func (SysAiTrace) TableName() string { return "sys_ai_trace" }

// SysAiLlmCall 单次 LLM 调用明细（span 级，raw_request/raw_response 为 wire 级原文）
type SysAiLlmCall struct {
	ID               int64           `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	TraceID          string          `gorm:"column:trace_id" json:"traceId"`
	Seq              int             `gorm:"column:seq" json:"seq"`
	StepPosition     *int            `gorm:"column:step_position" json:"stepPosition"`
	Model            *string         `gorm:"column:model" json:"model"`
	StartTime        *time.Time      `gorm:"column:start_time" json:"startTime"`
	Status           int             `gorm:"column:status" json:"status"`
	ErrorType        *string         `gorm:"column:error_type" json:"errorType"`
	DurationMs       int             `gorm:"column:duration_ms" json:"durationMs"`
	FirstTokenMs     *int            `gorm:"column:first_token_ms" json:"firstTokenMs"`
	PromptTokens     int             `gorm:"column:prompt_tokens" json:"promptTokens"`
	CompletionTokens int             `gorm:"column:completion_tokens" json:"completionTokens"`
	CachedTokens     int             `gorm:"column:cached_tokens" json:"cachedTokens"`
	ToolCall         json.RawMessage `gorm:"column:tool_call" json:"toolCall"`
	InputSnapshot    json.RawMessage `gorm:"column:input_snapshot" json:"inputSnapshot"`
	OutputSnapshot   json.RawMessage `gorm:"column:output_snapshot" json:"outputSnapshot"`
	Attempts         json.RawMessage `gorm:"column:attempts" json:"attempts"`
	RawRequest       json.RawMessage `gorm:"column:raw_request" json:"rawRequest"`
	RawResponse      json.RawMessage `gorm:"column:raw_response" json:"rawResponse"`
	CreateTime       time.Time       `gorm:"column:create_time;autoCreateTime" json:"createTime"`
}

func (SysAiLlmCall) TableName() string { return "sys_ai_llm_call" }
