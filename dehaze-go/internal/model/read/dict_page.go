package read

import "time"

// DictPage 字典分页读模型
type DictPage struct {
	ID         int64     `json:"id"`
	Name       string    `json:"name"`
	Value      string    `json:"value"`
	TypeCode   string    `json:"typeCode"`
	Defaulted  int8      `json:"defaulted"`
	Sort       int       `json:"sort"`
	Status     int8      `json:"status"`
	Remark     string    `json:"remark"`
	CreateTime time.Time `json:"createTime"`
}
