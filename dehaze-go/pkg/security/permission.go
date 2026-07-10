package security

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// 消息类型常量
const (
	CacheTypePermission = "permission"
	CacheTypeRole       = "role"
)

// cachedPerms 带过期时间的缓存权限数据
type cachedPerms struct {
	perms    []string
	expireAt time.Time
}

// PermissionChecker 权限检查器
// 修复说明 (P1 问题)：
// - 添加 cacheTTL 字段控制缓存过期时间
// - 使用 cachedPerms 结构体存储带过期时间的权限数据
// - 在读取缓存时检查过期时间
// - 默认缓存 TTL 为 5 分钟
// - 增加 Redis Pub/Sub 支持，实现多实例缓存失效广播
type PermissionChecker struct {
	cache      *sync.Map // 本地缓存，减少Redis查询
	mu         sync.RWMutex
	casbinImpl CasbinAdapter // Casbin适配器接口
	cacheTTL   time.Duration // 缓存过期时间
	pubsub     *redis.PubSub // Redis Pub/Sub 实例
}

// CasbinAdapter Casbin适配器接口，支持可选的Casbin集成
type CasbinAdapter interface {
	// Enforce 检查权限
	Enforce(sub, obj, act string) (bool, error)
	// BatchEnforce 批量检查权限
	BatchEnforce(requests [][]interface{}) (bool, error)
}

// permissionInstance 单例实例
var (
	permissionInstance *PermissionChecker
	permissionOnce     sync.Once
)

// GetPermissionChecker 获取权限检查器单例
// 默认缓存 TTL 为 5 分钟
func GetPermissionChecker() *PermissionChecker {
	permissionOnce.Do(func() {
		permissionInstance = &PermissionChecker{
			cache:    &sync.Map{},
			cacheTTL: 5 * time.Minute, // 默认缓存 5 分钟
		}
		// 尝试初始化 Pub/Sub
		permissionInstance.initPubSub()
	})
	return permissionInstance
}

// initPubSub 初始化 Pub/Sub 订阅
func (pc *PermissionChecker) initPubSub() {
	ps := redis.GetPubSub()
	if ps == nil {
		logger.Debug("Redis Pub/Sub 未启用，权限缓存失效广播不可用")
		return
	}

	pc.pubsub = ps

	// 订阅权限和角色缓存失效消息
	ps.Subscribe(CacheTypePermission, pc.handleCacheInvalidation)
	ps.Subscribe(CacheTypeRole, pc.handleCacheInvalidation)

	logger.Info("权限检查器已订阅缓存失效广播")
}

// handleCacheInvalidation 处理缓存失效消息
func (pc *PermissionChecker) handleCacheInvalidation(msg redis.CacheInvalidationMsg) {
	logger.Debug("处理缓存失效消息", zap.String("type", msg.Type), zap.String("key", msg.Key))

	switch msg.Type {
	case CacheTypePermission, CacheTypeRole:
		pc.deleteLocalCache(msg.Key)
	}
}

// deleteLocalCache 删除本地缓存
func (pc *PermissionChecker) deleteLocalCache(key string) {
	if key == "" {
		// 清理所有缓存
		pc.cache.Range(func(k, v interface{}) bool {
			pc.cache.Delete(k)
			return true
		})
		logger.Debug("已清理所有本地权限缓存")
		return
	}

	pc.cache.Delete(key)
	logger.Debug("已清理本地权限缓存", zap.String("key", key))
}

// SetCasbinAdapter 设置Casbin适配器（可选）
func (pc *PermissionChecker) SetCasbinAdapter(adapter CasbinAdapter) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.casbinImpl = adapter
}

// SetCacheTTL 设置缓存过期时间
func (pc *PermissionChecker) SetCacheTTL(ttl time.Duration) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.cacheTTL = ttl
}

// HasPermission 检查用户是否有指定权限（使用JWT中的权限列表）
// 参数：
//   - c: Gin上下文
//   - requiredPerm: 需要的权限标识
//
// 返回：
//   - bool: 是否有权限
//   - error: 错误信息
func HasPermission(c *gin.Context, requiredPerm string) (bool, error) {
	// 获取权限检查器实例
	pc := GetPermissionChecker()

	// 获取用户信息
	claims := GetUserInfo(c)
	if claims == nil {
		return false, fmt.Errorf("用户未登录")
	}

	// 如果配置了Casbin，使用Casbin检查
	if pc.casbinImpl != nil {
		userID := fmt.Sprintf("%d", claims.UserID)
		return pc.casbinImpl.Enforce(userID, requiredPerm, "*")
	}

	// 否则使用JWT中的权限列表检查
	return HasPermissionWithList(requiredPerm, claims.Authorities)
}

// HasPermissionWithWildcard 检查用户是否有权限，支持通配符
// 支持的通配符：
//   - *: 匹配任意多个字符（包括空字符）
//   - ?: 匹配单个字符
//
// 参数：
//   - c: Gin上下文
//   - requiredPerm: 需要的权限标识（可以包含通配符）
//
// 返回：
//   - bool: 是否有权限
//   - error: 错误信息
func HasPermissionWithWildcard(c *gin.Context, requiredPerm string) (bool, error) {
	// 获取权限检查器实例
	pc := GetPermissionChecker()

	// 获取用户信息
	claims := GetUserInfo(c)
	if claims == nil {
		return false, fmt.Errorf("用户未登录")
	}

	// 如果配置了Casbin，使用Casbin检查
	if pc.casbinImpl != nil {
		userID := fmt.Sprintf("%d", claims.UserID)
		return pc.casbinImpl.Enforce(userID, requiredPerm, "*")
	}

	// 否则使用JWT中的权限列表检查（支持通配符）
	return HasPermissionWithWildcardList(requiredPerm, claims.Authorities)
}

// HasPermissionWithList 使用权限列表检查权限
func HasPermissionWithList(requiredPerm string, userPerms []string) (bool, error) {
	// 遍历用户的所有权限
	for _, perm := range userPerms {
		if perm == requiredPerm {
			return true, nil
		}
	}
	return false, nil
}

// HasPermissionWithWildcardList 使用权限列表检查权限，支持通配符
// 支持的通配符：
//   - *: 匹配任意多个字符（包括空字符）
//   - ?: 匹配单个字符
//
// 匹配规则：
//  1. 用户权限为通配符模式，需要权限为具体值
//  2. 用户权限为具体值，需要权限为通配符模式
//
// 参数：
//   - requiredPerm: 需要的权限标识（可以包含通配符）
//   - userPerms: 用户拥有的权限列表
//
// 返回：
//   - bool: 是否有权限
//   - error: 错误信息
func HasPermissionWithWildcardList(requiredPerm string, userPerms []string) (bool, error) {
	// 如果需要权限为空，返回false
	if requiredPerm == "" {
		return false, nil
	}

	// 遍历用户的所有权限
	for _, userPerm := range userPerms {
		// 1. 精确匹配
		if userPerm == requiredPerm {
			return true, nil
		}

		// 2. 用户权限包含通配符，尝试匹配需要的权限
		if strings.ContainsAny(userPerm, "*?") {
			if matchWildcard(userPerm, requiredPerm) {
				return true, nil
			}
		}

		// 3. 需要权限包含通配符，尝试匹配用户的权限
		if strings.ContainsAny(requiredPerm, "*?") {
			if matchWildcard(requiredPerm, userPerm) {
				return true, nil
			}
		}
	}

	return false, nil
}

// matchWildcard 通配符匹配函数
// 支持的通配符：
//   - *: 匹配任意多个字符（包括空字符）
//   - ?: 匹配单个字符
//
// 参数：
//   - pattern: 包含通配符的模式
//   - str: 待匹配的字符串
//
// 返回：
//   - bool: 是否匹配
func matchWildcard(pattern, str string) bool {
	// 如果模式和字符串完全相同，直接返回true
	if pattern == str {
		return true
	}

	// 如果模式为"*"，匹配所有字符串
	if pattern == "*" {
		return true
	}

	// 动态规划匹配
	return wildcardMatch(pattern, str, 0, 0)
}

// wildcardMatch 通配符匹配的动态规划实现
func wildcardMatch(pattern, str string, pIdx, sIdx int) bool {
	// 如果模式和字符串都遍历完，匹配成功
	if pIdx == len(pattern) && sIdx == len(str) {
		return true
	}

	// 如果模式遍历完但字符串未遍历完，匹配失败
	if pIdx == len(pattern) {
		return false
	}

	// 如果模式当前字符是'*'
	if pattern[pIdx] == '*' {
		// *可以匹配0个字符或多个字符
		return wildcardMatch(pattern, str, pIdx+1, sIdx) || // 匹配0个字符
			(sIdx < len(str) && wildcardMatch(pattern, str, pIdx, sIdx+1)) // 匹配1个字符并继续匹配*
	}

	// 如果模式当前字符是'?'或者与字符串当前字符相同
	if sIdx < len(str) && (pattern[pIdx] == '?' || pattern[pIdx] == str[sIdx]) {
		return wildcardMatch(pattern, str, pIdx+1, sIdx+1)
	}

	// 其他情况匹配失败
	return false
}

// GetRolePermissions 从缓存或数据库获取角色权限
// 参数：
//   - roleCode: 角色编码
//
// 返回：
//   - []string: 权限列表
//   - error: 错误信息
func (pc *PermissionChecker) GetRolePermissions(roleCode string) ([]string, error) {
	now := time.Now()

	// 先从本地缓存获取，并检查是否过期
	if val, ok := pc.cache.Load("role:" + roleCode); ok {
		if cached, ok := val.(*cachedPerms); ok {
			if now.Before(cached.expireAt) {
				// 缓存未过期，直接返回
				return cached.perms, nil
			}
			// 缓存已过期，删除旧缓存
			pc.cache.Delete("role:" + roleCode)
		}
	}

	// 从缓存获取
	var perms []string
	cacheClient := cache.GetCache()
	result, err := cacheClient.HGet(context.Background(), "role_perms", roleCode)
	if err != nil {
		return nil, fmt.Errorf("获取角色权限失败: %w", err)
	}

	if result != "" {
		perms = strings.Split(result, ",")
	}

	// 写入本地缓存（带过期时间）
	if len(perms) > 0 {
		pc.cache.Store("role:"+roleCode, &cachedPerms{
			perms:    perms,
			expireAt: now.Add(pc.cacheTTL),
		})
	}

	return perms, nil
}

// ClearRolePermissionCache 清理角色权限缓存
// 同时广播到其他实例
// 参数：
//   - roleCode: 角色编码，为空则清理所有角色缓存
func (pc *PermissionChecker) ClearRolePermissionCache(roleCode string) {
	cacheKey := ""
	if roleCode != "" {
		cacheKey = "role:" + roleCode
		pc.cache.Delete(cacheKey)
	} else {
		// 清理所有角色权限缓存
		pc.cache.Range(func(key, value interface{}) bool {
			if k, ok := key.(string); ok && strings.HasPrefix(k, "role:") {
				pc.cache.Delete(k)
			}
			return true
		})
	}

	// 广播缓存失效消息
	pc.broadcastInvalidation(CacheTypeRole, cacheKey)
}

// broadcastInvalidation 广播缓存失效消息
func (pc *PermissionChecker) broadcastInvalidation(msgType, key string) {
	if pc.pubsub == nil {
		return
	}

	ctx := context.Background()
	if err := pc.pubsub.Publish(ctx, msgType, key); err != nil {
		logger.Warn("广播缓存失效消息失败",
			zap.String("type", msgType),
			zap.String("key", key),
			zap.Error(err),
		)
	}
}

// ClearUserPermissionCache 清理用户权限缓存
// 同时广播到其他实例
func (pc *PermissionChecker) ClearUserPermissionCache(userID string) {
	cacheKey := ""
	if userID != "" {
		cacheKey = "user:" + userID
		pc.cache.Delete(cacheKey)
	} else {
		// 清理所有用户权限缓存
		pc.cache.Range(func(key, value interface{}) bool {
			if k, ok := key.(string); ok && strings.HasPrefix(k, "user:") {
				pc.cache.Delete(k)
			}
			return true
		})
	}

	// 广播缓存失效消息
	pc.broadcastInvalidation(CacheTypePermission, cacheKey)
}

// HasRolePermission 检查用户角色是否有指定权限（支持通配符）
// 参数：
//   - roleCode: 角色编码
//   - requiredPerm: 需要的权限标识（可以包含通配符）
//
// 返回：
//   - bool: 是否有权限
//   - error: 错误信息
func (pc *PermissionChecker) HasRolePermission(roleCode, requiredPerm string) (bool, error) {
	// 获取角色权限
	perms, err := pc.GetRolePermissions(roleCode)
	if err != nil {
		return false, err
	}

	// 使用通配符匹配
	return HasPermissionWithWildcardList(requiredPerm, perms)
}

// BatchHasPermission 批量检查权限
// 参数：
//   - c: Gin上下文
//   - requiredPerms: 需要的权限列表
//   - requireAll: 是否需要满足所有权限（true:全部满足，false:任一满足）
//
// 返回：
//   - bool: 是否有权限
//   - error: 错误信息
func BatchHasPermission(c *gin.Context, requiredPerms []string, requireAll bool) (bool, error) {
	if len(requiredPerms) == 0 {
		return true, nil
	}

	// 获取用户信息
	claims := GetUserInfo(c)
	if claims == nil {
		return false, fmt.Errorf("用户未登录")
	}

	if requireAll {
		// 需要满足所有权限
		for _, perm := range requiredPerms {
			hasPerm, err := HasPermissionWithWildcardList(perm, claims.Authorities)
			if err != nil {
				return false, err
			}
			if !hasPerm {
				return false, nil
			}
		}
		return true, nil
	} else {
		// 需要满足任一权限
		for _, perm := range requiredPerms {
			hasPerm, err := HasPermissionWithWildcardList(perm, claims.Authorities)
			if err != nil {
				return false, err
			}
			if hasPerm {
				return true, nil
			}
		}
		return false, nil
	}
}

// HasAnyPermission 检查用户是否有任一权限
// 参数：
//   - c: Gin上下文
//   - perms: 权限列表
//
// 返回：
//   - bool: 是否有任一权限
//   - error: 错误信息
func HasAnyPermission(c *gin.Context, perms ...string) (bool, error) {
	return BatchHasPermission(c, perms, false)
}

// HasAllPermissions 检查用户是否有所有权限
// 参数：
//   - c: Gin上下文
//   - perms: 权限列表
//
// 返回：
//   - bool: 是否有所有权限
//   - error: 错误信息
func HasAllPermissions(c *gin.Context, perms ...string) (bool, error) {
	return BatchHasPermission(c, perms, true)
}
