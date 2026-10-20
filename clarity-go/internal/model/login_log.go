package model

import (
	"time"

	"go.mongodb.org/mongo-driver/bson/primitive"
)

// LoginLog 登录审计日志（MongoDB document: login_log）
// bson 字段名与 Python 端 LoginLogDocument 对齐（三端共享同一集合）
type LoginLog struct {
	ID         primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	UserID     *int64             `bson:"user_id,omitempty" json:"userId"`
	Username   string             `bson:"username" json:"username"`
	IP         string             `bson:"ip" json:"ip"`
	Location   string             `bson:"location" json:"location"`
	Browser    string             `bson:"browser" json:"browser"`
	OS         string             `bson:"os" json:"os"`
	DeviceType string             `bson:"device_type" json:"deviceType"`
	Status     int                `bson:"status" json:"status"`
	Message    string             `bson:"message" json:"message"`
	CreateTime time.Time          `bson:"create_time" json:"createTime"`
}

func (LoginLog) CollectionName() string {
	return "login_log"
}
