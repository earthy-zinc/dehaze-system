package model

import "time"

type LoginLog struct {
	UserID      *int64    `bson:"userId,omitempty" json:"userId"`
	Username    string    `bson:"username" json:"username"`
	IP          string    `bson:"ip" json:"ip"`
	Location    string    `bson:"location" json:"location"`
	Browser     string    `bson:"browser" json:"browser"`
	OS          string    `bson:"os" json:"os"`
	Status      int       `bson:"status" json:"status"`
	Message     string    `bson:"message" json:"message"`
	CreateTime  time.Time `bson:"createTime" json:"createTime"`
}

func (LoginLog) CollectionName() string {
	return "login_log"
}
