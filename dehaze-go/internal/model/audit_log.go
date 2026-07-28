package model

import "time"

type AuditLog struct {
	OperatorID  int64       `bson:"operatorId" json:"operatorId"`
	TargetType  string      `bson:"targetType" json:"targetType"`
	TargetID    interface{} `bson:"targetId,omitempty" json:"targetId"`
	Action      string      `bson:"action" json:"action"`
	Module      string      `bson:"module" json:"module"`
	BeforeValue interface{} `bson:"beforeValue,omitempty" json:"beforeValue"`
	AfterValue  interface{} `bson:"afterValue,omitempty" json:"afterValue"`
	IP          string      `bson:"ip" json:"ip"`
	UserAgent   string      `bson:"userAgent" json:"userAgent"`
	CreateTime  time.Time   `bson:"createTime" json:"createTime"`
}

func (AuditLog) CollectionName() string {
	return "audit_log"
}
