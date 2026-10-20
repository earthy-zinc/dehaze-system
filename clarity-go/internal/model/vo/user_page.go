package vo

// UserPageVO 用户分页视图对象
type UserPageVO struct {
	// 用户ID
	ID int64 `json:"id"`
	// 用户名
	Username string `json:"username"`
	// 用户昵称
	Nickname string `json:"nickname"`
	// 手机号
	Mobile string `json:"mobile"`
	// 性别
	GenderLabel string `json:"genderLabel"`
	// 用户头像地址
	Avatar string `json:"avatar"`
	// 用户邮箱
	Email string `json:"email"`
	// 用户状态(1:启用;0:禁用)
	Status int8 `json:"status"`
	// 用户类型(personal:个人;enterprise:企业)
	UserType string `json:"userType"`
	// 会员等级(level_0~level_3)，无会员记录为 null
	MemberLevel *string `json:"memberLevel"`
	// 会员套餐到期时间（yyyy-MM-dd HH:mm:ss，与 Java/Python 统一格式），null 表示成长值维持
	MemberExpireTime *string `json:"memberExpireTime"`
	// 本月配额用量（used/total，8 类任务求和），无会员记录为 "0/0"
	QuotaUsage string `json:"quotaUsage"`
	// 部门名称
	DeptName string `json:"deptName"`
	// 角色名称，多个使用英文逗号(,)分割
	RoleNames string `json:"roleNames"`
	// 创建时间
	CreateTime string `json:"createTime"`
}
