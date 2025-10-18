package query

// UserPageQuery 用户分页查询对象
type UserPageQuery struct {
	// 关键字(用户名/昵称/手机号)
	Keywords string `json:"keywords"`
	// 用户状态
	Status *int `json:"status"`
	// 部门ID
	DeptId *int64 `json:"deptId"`
	// 创建时间-开始时间
	StartTime string `json:"startTime"`
	// 创建时间-结束时间
	EndTime string `json:"endTime"`
	// 页码
	PageNum int `json:"pageNum"`
	// 每页条数
	PageSize int `json:"pageSize"`
}