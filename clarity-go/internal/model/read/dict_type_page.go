package read

import "time"

// DictTypePage 字典类型分页读模型
type DictTypePage struct {
	ID         int64     `json:"id"`
	Name       string    `json:"name"`
	Code       string    `json:"code"`
	Status     int8      `json:"status"`
	Remark     string    `json:"remark"`
	CreateTime time.Time `json:"createTime"`
}
