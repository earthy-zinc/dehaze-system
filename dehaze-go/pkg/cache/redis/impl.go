package redis

import (
	"context"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache/errs"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

const (
	defaultLockPrefix = "lock:"
)

type RedisCache struct {
	client *redis.Client
}

func NewRedisCache(client *redis.Client) *RedisCache {
	return &RedisCache{client: client}
}

func (c *RedisCache) Get(ctx context.Context, key string) (string, error) {
	val, err := c.client.Get(ctx, key).Result()
	if err != nil {
		if err == redis.Nil {
			return "", errs.ErrKeyNotFound
		}
		return "", err
	}
	return val, nil
}

func (c *RedisCache) Set(ctx context.Context, key string, value any, expiration time.Duration) error {
	return c.client.Set(ctx, key, value, expiration).Err()
}

func (c *RedisCache) Delete(ctx context.Context, keys ...string) error {
	return c.client.Del(ctx, keys...).Err()
}

func (c *RedisCache) Exists(ctx context.Context, key string) (bool, error) {
	n, err := c.client.Exists(ctx, key).Result()
	return n > 0, err
}

func (c *RedisCache) SetNX(ctx context.Context, key string, value any, expiration time.Duration) (bool, error) {
	return c.client.SetNX(ctx, key, value, expiration).Result()
}

func (c *RedisCache) MGet(ctx context.Context, keys ...string) ([]string, error) {
	if len(keys) == 0 {
		return []string{}, nil
	}
	vals, err := c.client.MGet(ctx, keys...).Result()
	if err != nil {
		return nil, err
	}

	result := make([]string, len(vals))
	for i, val := range vals {
		if val == nil {
			result[i] = ""
		} else {
			result[i] = fmt.Sprintf("%v", val)
		}
	}
	return result, nil
}

func (c *RedisCache) MSet(ctx context.Context, kvPairs map[string]any, expiration time.Duration) error {
	if len(kvPairs) == 0 {
		return nil
	}

	// 如果没有过期时间，直接使用MSet（更高效）
	if expiration == 0 {
		pipe := c.client.Pipeline()
		for key, value := range kvPairs {
			pipe.Set(ctx, key, value, 0)
		}
		_, err := pipe.Exec(ctx)
		return err
	}

	// 有过期时间，需要逐个设置
	pipe := c.client.Pipeline()
	for key, value := range kvPairs {
		pipe.Set(ctx, key, value, expiration)
	}
	_, err := pipe.Exec(ctx)
	return err
}

func (c *RedisCache) MDelete(ctx context.Context, keys ...string) error {
	if len(keys) == 0 {
		return nil
	}
	return c.client.Del(ctx, keys...).Err()
}

func (c *RedisCache) Incr(ctx context.Context, key string) (int64, error) {
	return c.client.Incr(ctx, key).Result()
}

func (c *RedisCache) IncrBy(ctx context.Context, key string, value int64) (int64, error) {
	return c.client.IncrBy(ctx, key, value).Result()
}

func (c *RedisCache) Decr(ctx context.Context, key string) (int64, error) {
	return c.client.Decr(ctx, key).Result()
}

func (c *RedisCache) DecrBy(ctx context.Context, key string, value int64) (int64, error) {
	return c.client.DecrBy(ctx, key, value).Result()
}

func (c *RedisCache) Expire(ctx context.Context, key string, expiration time.Duration) (bool, error) {
	return c.client.Expire(ctx, key, expiration).Result()
}

func (c *RedisCache) TTL(ctx context.Context, key string) (time.Duration, error) {
	return c.client.TTL(ctx, key).Result()
}

// unlockScript Lua 脚本：仅当锁的持有者匹配时才删除锁
const unlockScript = `
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end`

func (c *RedisCache) Lock(ctx context.Context, key string, expiration time.Duration) (string, bool, error) {
	lockKey := fmt.Sprintf("%s%s", defaultLockPrefix, key)
	token := uuid.New().String()
	ok, err := c.client.SetNX(ctx, lockKey, token, expiration).Result()
	if err != nil {
		logger.Error("获取分布式锁失败", zap.String("key", lockKey), zap.Error(err))
		return "", false, err
	}
	return token, ok, nil
}

func (c *RedisCache) Unlock(ctx context.Context, key string, token string) (bool, error) {
	lockKey := fmt.Sprintf("%s%s", defaultLockPrefix, key)
	result, err := c.client.Eval(ctx, unlockScript, []string{lockKey}, token).Result()
	if err != nil {
		logger.Error("释放分布式锁失败", zap.String("key", lockKey), zap.Error(err))
		return false, err
	}
	// Eval 返回 int64 类型（DEL 返回删除的 key 数量）
	count, _ := result.(int64)
	return count > 0, nil
}

func (c *RedisCache) Pipeline(ctx context.Context, ops []types.PipelineOp) error {
	if len(ops) == 0 {
		return nil
	}

	pipe := c.client.Pipeline()
	for _, op := range ops {
		switch op.Type {
		case "set":
			pipe.Set(ctx, op.Key, op.Value, op.Exp)
		case "delete":
			pipe.Del(ctx, op.Key)
		case "incr":
			pipe.Incr(ctx, op.Key)
		case "decr":
			pipe.Decr(ctx, op.Key)
		}
	}

	_, err := pipe.Exec(ctx)
	return err
}

// HGet 获取哈希表中指定字段的值
func (c *RedisCache) HGet(ctx context.Context, key, field string) (string, error) {
	val, err := c.client.HGet(ctx, key, field).Result()
	if err != nil {
		if err == redis.Nil {
			return "", errs.ErrKeyNotFound
		}
		return "", err
	}
	return val, nil
}

// HSet 设置哈希表中指定字段的值
func (c *RedisCache) HSet(ctx context.Context, key, field string, value any) error {
	return c.client.HSet(ctx, key, field, value).Err()
}

// HDel 删除哈希表中的一个或多个字段
func (c *RedisCache) HDel(ctx context.Context, key string, fields ...string) error {
	if len(fields) == 0 {
		return nil
	}
	return c.client.HDel(ctx, key, fields...).Err()
}

// HGetAll 获取哈希表中所有字段和值
func (c *RedisCache) HGetAll(ctx context.Context, key string) (map[string]string, error) {
	return c.client.HGetAll(ctx, key).Result()
}

// Set 操作

// SAdd 向集合添加一个或多个成员
func (c *RedisCache) SAdd(ctx context.Context, key string, members ...any) error {
	if len(members) == 0 {
		return nil
	}
	return c.client.SAdd(ctx, key, members...).Err()
}

// SMembers 返回集合中的所有成员
func (c *RedisCache) SMembers(ctx context.Context, key string) ([]string, error) {
	return c.client.SMembers(ctx, key).Result()
}

// SRem 移除集合中一个或多个成员
func (c *RedisCache) SRem(ctx context.Context, key string, members ...any) error {
	if len(members) == 0 {
		return nil
	}
	return c.client.SRem(ctx, key, members...).Err()
}
