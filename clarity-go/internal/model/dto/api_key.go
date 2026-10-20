package dto

import "time"

type ApiKeyCreateRequest struct {
	Name      string     `json:"name" binding:"required"`
	ExpiresAt *time.Time `json:"expiresAt"`
	// 治理参数：不传即不限制（python ApiKeyCreate 的 ge=1 校验同口径）
	DailyQuota   *int64 `json:"dailyQuota" binding:"omitempty,min=1"`
	MonthlyQuota *int64 `json:"monthlyQuota" binding:"omitempty,min=1"`
	RpmLimit     *int   `json:"rpmLimit" binding:"omitempty,min=1"`
	// 模型白名单：不传或空数组 = 继承用户可见模型（python ApiKeyCreate.modelWhitelist 同口径）
	ModelWhitelist []string `json:"modelWhitelist"`
}

type ApiKeyResult struct {
	ID         int64      `json:"id"`
	Name       string     `json:"name"`
	ApiKey     string     `json:"apiKey,omitempty"`
	KeyPrefix  string     `json:"keyPrefix"`
	Status     int8       `json:"status"`
	ExpiresAt  *time.Time `json:"expiresAt"`
	LastUsedAt *time.Time `json:"lastUsedAt"`
	CreateTime time.Time  `json:"createTime"`
	// 治理参数回传（python ApiKeyResult 同形态；NULL 表示不限制，null 字段整键省略）
	DailyQuota     *int64   `json:"dailyQuota"`
	MonthlyQuota   *int64   `json:"monthlyQuota"`
	RpmLimit       *int     `json:"rpmLimit"`
	ModelWhitelist []string `json:"modelWhitelist,omitempty"`
}
