package model

import (
	"time"

	"go.mongodb.org/mongo-driver/bson/primitive"
)

// AuditLog 审计日志（MongoDB 集合 audit_log）。
//
// bson 键名一律 snake_case，与 python 写入端逐字一致（`mongo_audit_log_repository.create_audit`：
// operator_id/target_type/target_id/action/module/before_value/after_value/ip/user_agent/create_time）。
// 注意：这里必须是 snake_case —— 两边若不一致，go 写、python 读会静默全 None（反之亦然）；
// json 标签仍为 camelCase，供 API 出参使用。
type AuditLog struct {
	ID          primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	OperatorID  int64              `bson:"operator_id" json:"operatorId"`
	TargetType  string             `bson:"target_type" json:"targetType"`
	TargetID    interface{}        `bson:"target_id,omitempty" json:"targetId"`
	Action      string             `bson:"action" json:"action"`
	Module      string             `bson:"module" json:"module"`
	BeforeValue interface{}        `bson:"before_value,omitempty" json:"beforeValue"`
	AfterValue  interface{}        `bson:"after_value,omitempty" json:"afterValue"`
	IP          string             `bson:"ip" json:"ip"`
	UserAgent   string             `bson:"user_agent" json:"userAgent"`
	CreateTime  time.Time          `bson:"create_time" json:"createTime"`
}

func (AuditLog) CollectionName() string {
	return "audit_log"
}
