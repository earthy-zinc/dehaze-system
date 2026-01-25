package types

import (
	"context"
	"time"
)

// PipelineOp 定义Pipeline操作类型
type PipelineOp struct {
	Type  string // "get", "set", "delete", "incr", "decr"
	Key   string
	Value any
	Exp   time.Duration
}

type ICache interface {
	Get(ctx context.Context, key string) (string, error)
	Set(ctx context.Context, key string, value any, expiration time.Duration) error
	Delete(ctx context.Context, keys ...string) error
	Exists(ctx context.Context, key string) (bool, error)
	SetNX(ctx context.Context, key string, value any, expiration time.Duration) (bool, error)

	// MGet 批量获取多个key的值
	MGet(ctx context.Context, keys ...string) ([]string, error)
	// MSet 批量设置多个key-value对
	MSet(ctx context.Context, kvPairs map[string]any, expiration time.Duration) error
	// MDelete 批量删除多个key
	MDelete(ctx context.Context, keys ...string) error

	// Incr 将key中的数值加1
	Incr(ctx context.Context, key string) (int64, error)
	// IncrBy 将key中的数值增加指定值
	IncrBy(ctx context.Context, key string, value int64) (int64, error)
	// Decr 将key中的数值减1
	Decr(ctx context.Context, key string) (int64, error)
	// DecrBy 将key中的数值减少指定值
	DecrBy(ctx context.Context, key string, value int64) (int64, error)

	// Expire 设置key的过期时间
	Expire(ctx context.Context, key string, expiration time.Duration) (bool, error)
	// TTL 获取key的剩余过期时间
	TTL(ctx context.Context, key string) (time.Duration, error)

	// Lock 尝试获取分布式锁
	Lock(ctx context.Context, key string, expiration time.Duration) (bool, error)
	// Unlock 释放分布式锁
	Unlock(ctx context.Context, key string) (bool, error)

	// Hash 操作
	// HGet 获取哈希表中指定字段的值
	HGet(ctx context.Context, key, field string) (string, error)
	// HSet 设置哈希表中指定字段的值
	HSet(ctx context.Context, key, field string, value any) error
	// HDel 删除哈希表中的一个或多个字段
	HDel(ctx context.Context, key string, fields ...string) error
	// HGetAll 获取哈希表中所有字段和值
	HGetAll(ctx context.Context, key string) (map[string]string, error)

	// Set 操作
	// SAdd 向集合添加一个或多个成员
	SAdd(ctx context.Context, key string, members ...any) error
	// SMembers 返回集合中的所有成员
	SMembers(ctx context.Context, key string) ([]string, error)
	// SRem 移除集合中一个或多个成员
	SRem(ctx context.Context, key string, members ...any) error

	// Pipeline 批量执行Pipeline操作
	Pipeline(ctx context.Context, ops []PipelineOp) error
}
