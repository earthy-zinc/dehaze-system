package dto

import "time"

type ApiKeyCreateRequest struct {
	Name      string     `json:"name" binding:"required"`
	ExpiresAt *time.Time `json:"expiresAt"`
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
}
