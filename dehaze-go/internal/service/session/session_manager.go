// Package session 提供在线会话的查询与踢出能力（F-AM-011）。
// 独立于 AuthService 存在：用户模块禁用/删除/重置密码时需要踢出目标用户会话，
// 而 auth 包依赖 user 包（IUserService），反向引用会构成循环依赖。
package session

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"time"

	goredis "github.com/redis/go-redis/v9"

	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
)

// AdminMaxDevices 管理员（ROOT/ADMIN）不受等级权益约束，固定 10 台
const AdminMaxDevices = 10

// SessionInfo 在线会话信息（响应 VO，字段与 Python 端 list_sessions 一致）
type SessionInfo struct {
	SessionID      string `json:"sessionId"`
	Username       string `json:"username"`
	DeviceType     string `json:"deviceType"`
	LoginTime      string `json:"loginTime"`
	IP             string `json:"ip"`
	LastAccessTime string `json:"lastAccessTime"`
}

// ListByUsername 扫描 session:* 并按用户名精确过滤（排除多点登录索引键）。
func ListByUsername(ctx context.Context, username string) ([]SessionInfo, error) {
	client := redis.GetClient()
	if client == nil {
		return nil, common.NewBizError(common.CACHE_SERVICE_ERROR, "缓存服务不可用")
	}

	sessions := make([]SessionInfo, 0)
	iter := client.Scan(ctx, 0, common.SessionPrefix+"*", 100).Iterator()
	for iter.Next(ctx) {
		key := iter.Val()
		if strings.HasPrefix(key, common.SessionUserPrefix) {
			continue
		}
		raw, err := client.Get(ctx, key).Result()
		if err != nil {
			continue
		}
		var data middleware.SessionData
		if err := json.Unmarshal([]byte(raw), &data); err != nil {
			continue
		}
		if data.Username != username {
			continue
		}
		deviceType := data.DeviceType
		if deviceType == "" {
			deviceType = "web"
		}
		sessions = append(sessions, SessionInfo{
			SessionID:      strings.TrimPrefix(key, common.SessionPrefix),
			Username:       data.Username,
			DeviceType:     deviceType,
			LoginTime:      data.LoginTime,
			IP:             data.LoginIP,
			LastAccessTime: data.LastAccessTime,
		})
	}
	if err := iter.Err(); err != nil {
		return nil, common.WrapBizError(common.CACHE_SERVICE_ERROR, "扫描在线会话失败", err)
	}
	return sessions, nil
}

// KickByID 踢出指定会话。会话不存在返回 A0401，超级管理员会话不可踢出（A0503）。
func KickByID(ctx context.Context, sessionID string) error {
	client := redis.GetClient()
	if client == nil {
		return common.NewBizError(common.CACHE_SERVICE_ERROR, "缓存服务不可用")
	}

	key := common.SessionPrefix + sessionID
	raw, err := client.Get(ctx, key).Result()
	if err != nil || raw == "" {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在或已过期")
	}
	var data middleware.SessionData
	if err := json.Unmarshal([]byte(raw), &data); err != nil {
		return common.WrapBizError(common.CACHE_SERVICE_ERROR, "解析会话数据失败", err)
	}
	for _, authority := range data.Authorities {
		if authority == "ROLE_ROOT" {
			return common.NewBizError(common.OPERATION_NOT_ALLOW, "超级管理员会话不可被踢出")
		}
	}
	if err := client.Del(ctx, key).Err(); err != nil {
		return common.WrapBizError(common.CACHE_SERVICE_ERROR, "删除会话失败", err)
	}
	return nil
}

// KickByUserIDs 踢出一批用户的全部在线会话（单次扫描），返回踢出的会话数。
// 超级管理员会话不可被踢出，与 KickByID 语义一致。
func KickByUserIDs(ctx context.Context, userIDs []int64) (int, error) {
	if len(userIDs) == 0 {
		return 0, nil
	}
	client := redis.GetClient()
	if client == nil {
		return 0, common.NewBizError(common.CACHE_SERVICE_ERROR, "缓存服务不可用")
	}

	targets := make(map[int64]struct{}, len(userIDs))
	for _, id := range userIDs {
		targets[id] = struct{}{}
	}

	kicked := 0
	var keys []string
	kickedUsers := make(map[int64]struct{})
	iter := client.Scan(ctx, 0, common.SessionPrefix+"*", 100).Iterator()
	for iter.Next(ctx) {
		key := iter.Val()
		if strings.HasPrefix(key, common.SessionUserPrefix) {
			continue
		}
		raw, err := client.Get(ctx, key).Result()
		if err != nil {
			continue
		}
		var data middleware.SessionData
		if err := json.Unmarshal([]byte(raw), &data); err != nil {
			continue
		}
		if _, ok := targets[data.UserID]; ok {
			keys = append(keys, key)
			kickedUsers[data.UserID] = struct{}{}
		}
	}
	if err := iter.Err(); err != nil {
		return kicked, common.WrapBizError(common.CACHE_SERVICE_ERROR, "扫描在线会话失败", err)
	}
	if len(keys) > 0 {
		if err := client.Del(ctx, keys...).Err(); err != nil {
			return kicked, common.WrapBizError(common.CACHE_SERVICE_ERROR, "删除会话失败", err)
		}
		kicked = len(keys)
	}
	for userID := range kickedUsers {
		if err := client.Del(ctx, common.SessionUserPrefix+strconv.FormatInt(userID, 10)).Err(); err != nil {
			return kicked, common.WrapBizError(common.CACHE_SERVICE_ERROR, "清理会话索引失败", err)
		}
	}
	return kicked, nil
}

// RegisterSession 将会话登记进用户会话索引，并按同时在线设备数上限踢出最早登录的会话（F-AM-011）。
//
// 索引 session:user:{userId} 为 ZSet（member=sessionId，score=登录 epoch 秒，截断到秒与
// 本项目 DATETIME 秒精度口径一致），三端共享同一 Redis 结构。超限时新会话保留，最早的
// 若干会话被删除（session:{sessionId} + 索引元素），其下一次请求因会话不存在而返回 401。
func RegisterSession(ctx context.Context, userID int64, sessionID string, maxDevices int) error {
	client := redis.GetClient()
	if client == nil {
		return common.NewBizError(common.CACHE_SERVICE_ERROR, "缓存服务不可用")
	}

	indexKey := common.SessionUserPrefix + strconv.FormatInt(userID, 10)
	score := float64(time.Now().Truncate(time.Second).Unix())
	if err := client.ZAdd(ctx, indexKey, goredis.Z{Score: score, Member: sessionID}).Err(); err != nil {
		return common.WrapBizError(common.CACHE_SERVICE_ERROR, "写入会话索引失败", err)
	}
	if err := client.Expire(ctx, indexKey, middleware.SessionTTL).Err(); err != nil {
		return common.WrapBizError(common.CACHE_SERVICE_ERROR, "设置会话索引过期时间失败", err)
	}

	total, err := client.ZCard(ctx, indexKey).Result()
	if err != nil {
		return common.WrapBizError(common.CACHE_SERVICE_ERROR, "读取在线设备数失败", err)
	}
	if total <= int64(maxDevices) {
		return nil
	}

	members, err := client.ZRange(ctx, indexKey, 0, -1).Result()
	if err != nil {
		return common.WrapBizError(common.CACHE_SERVICE_ERROR, "读取在线会话失败", err)
	}
	excess := int(total) - maxDevices
	evicted := make([]string, 0, excess)
	for _, sid := range members {
		// 排除本次新会话：同秒登录时 score 相同，按 member 字典序排序也可能把它排在前面
		if sid == sessionID {
			continue
		}
		evicted = append(evicted, sid)
		if len(evicted) == excess {
			break
		}
	}
	if len(evicted) == 0 {
		return nil
	}

	keys := make([]string, 0, len(evicted))
	remArgs := make([]interface{}, 0, len(evicted))
	for _, sid := range evicted {
		keys = append(keys, common.SessionPrefix+sid)
		remArgs = append(remArgs, sid)
	}
	if err := client.Del(ctx, keys...).Err(); err != nil {
		return common.WrapBizError(common.CACHE_SERVICE_ERROR, "踢出超限会话失败", err)
	}
	if err := client.ZRem(ctx, indexKey, remArgs...).Err(); err != nil {
		return common.WrapBizError(common.CACHE_SERVICE_ERROR, "清理会话索引失败", err)
	}
	return nil
}

// RemoveFromIndex 从用户会话索引移除指定会话元素（注销时调用，不影响该用户其他会话）。
func RemoveFromIndex(ctx context.Context, userID int64, sessionID string) error {
	client := redis.GetClient()
	if client == nil {
		return common.NewBizError(common.CACHE_SERVICE_ERROR, "缓存服务不可用")
	}
	indexKey := common.SessionUserPrefix + strconv.FormatInt(userID, 10)
	if err := client.ZRem(ctx, indexKey, sessionID).Err(); err != nil {
		return common.WrapBizError(common.CACHE_SERVICE_ERROR, "清理会话索引失败", err)
	}
	return nil
}
