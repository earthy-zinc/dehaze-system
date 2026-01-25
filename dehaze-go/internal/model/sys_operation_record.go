package model

import (
	"time"
)

// SysOperationRecord 操作记录表
type SysOperationRecord struct {
	BaseModel
	Ip           string        `json:"ip" form:"ip" gorm:"column:ip;type:varchar(255);comment:请求ip"`
	Method       string        `json:"method" form:"method" gorm:"column:method;type:varchar(50);comment:请求方法"`
	Path         string        `json:"path" form:"path" gorm:"column:path;type:varchar(255);comment:请求路径"`
	Status       int           `json:"status" form:"status" gorm:"column:status;type:int;comment:请求状态"`
	Latency      time.Duration `json:"latency" form:"latency" gorm:"column:latency;type:bigint;comment:延迟(纳秒)" swaggertype:"string"`
	Agent        string        `json:"agent" form:"agent" gorm:"column:agent;type:text;comment:代理"`
	ErrorMessage string        `json:"error_message" form:"error_message" gorm:"column:error_message;type:varchar(512);comment:错误信息"`
	Body         string        `json:"body" form:"body" gorm:"column:body;type:text;comment:请求Body"`
	Resp         string        `json:"resp" form:"resp" gorm:"column:resp;type:text;comment:响应Body"`
	UserID       int64         `json:"user_id" form:"user_id" gorm:"column:user_id;type:bigint;comment:用户id"`
	User         SysUser       `json:"user" gorm:"foreignKey:UserID"`
}
