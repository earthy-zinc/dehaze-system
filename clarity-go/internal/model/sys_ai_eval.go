package model

import "time"

// SysAiAgentEvalDataset 评测集
type SysAiAgentEvalDataset struct {
	ID          int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	AgentID     int64      `gorm:"column:agent_id" json:"agentId"`
	Name        string     `gorm:"column:name" json:"name"`
	Description string     `gorm:"column:description" json:"description"`
	DatasetType string     `gorm:"column:dataset_type" json:"datasetType"`
	Deleted     int64      `gorm:"column:deleted" json:"deleted"`
	CreateBy    *int64     `gorm:"column:create_by" json:"createBy"`
	UpdateBy    *int64     `gorm:"column:update_by" json:"updateBy"`
	CreateTime  time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime  *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiAgentEvalDataset) TableName() string { return "sys_ai_agent_eval_dataset" }

// SysAiAgentEvalSample 评测样本（随评测集管理，无逻辑删除）
type SysAiAgentEvalSample struct {
	ID                int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	DatasetID         int64      `gorm:"column:dataset_id" json:"datasetId"`
	TaskGoal          string     `gorm:"column:task_goal" json:"taskGoal"`
	AllowedInput      *string    `gorm:"column:allowed_input" json:"allowedInput"`
	Tools             string     `gorm:"column:tools;type:json;default:null" json:"tools"`
	ExpectedProcess   *string    `gorm:"column:expected_process" json:"expectedProcess"`
	ExpectedResult    *string    `gorm:"column:expected_result" json:"expectedResult"`
	ForbiddenBehavior *string    `gorm:"column:forbidden_behavior" json:"forbiddenBehavior"`
	RiskLevel         string     `gorm:"column:risk_level" json:"riskLevel"`
	CreateBy          *int64     `gorm:"column:create_by" json:"createBy"`
	UpdateBy          *int64     `gorm:"column:update_by" json:"updateBy"`
	CreateTime        time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime        *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiAgentEvalSample) TableName() string { return "sys_ai_agent_eval_sample" }

// SysAiAgentEvalRun 评测执行记录（只追加）
type SysAiAgentEvalRun struct {
	ID           int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	AgentID      int64      `gorm:"column:agent_id" json:"agentId"`
	DatasetID    int64      `gorm:"column:dataset_id" json:"datasetId"`
	TriggerType  string     `gorm:"column:trigger_type" json:"triggerType"`
	Status       int        `gorm:"column:status" json:"status"`
	ScoreSummary string     `gorm:"column:score_summary;type:json;default:null" json:"scoreSummary"`
	Results      string     `gorm:"column:results;type:json;default:null" json:"results"`
	CreateBy     *int64     `gorm:"column:create_by" json:"createBy"`
	CreateTime   time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime   *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiAgentEvalRun) TableName() string { return "sys_ai_agent_eval_run" }

// SysAiEvalReview 评测人工复核
type SysAiEvalReview struct {
	ID          int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	RunID       int64      `gorm:"column:run_id" json:"runId"`
	SampleID    int64      `gorm:"column:sample_id" json:"sampleId"`
	AgentID     int64      `gorm:"column:agent_id" json:"agentId"`
	JudgePassed int        `gorm:"column:judge_passed" json:"judgePassed"`
	RiskLevel   string     `gorm:"column:risk_level" json:"riskLevel"`
	Status      int        `gorm:"column:status" json:"status"`
	Agree       *int       `gorm:"column:agree" json:"agree"`
	ReviewerID  *int64     `gorm:"column:reviewer_id" json:"reviewerId"`
	Remark      *string    `gorm:"column:remark" json:"remark"`
	CreateBy    *int64     `gorm:"column:create_by" json:"createBy"`
	CreateTime  time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateBy    *int64     `gorm:"column:update_by" json:"updateBy"`
	UpdateTime  *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiEvalReview) TableName() string { return "sys_ai_eval_review" }
