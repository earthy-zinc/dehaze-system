package security

import (
	"fmt"
	"strings"

	"github.com/gin-gonic/gin"
)

// HasPermission 检查用户是否有指定权限（使用JWT中的权限列表）
// 参数：
//   - c: Gin上下文
//   - requiredPerm: 需要的权限标识
//
// 返回：
//   - bool: 是否有权限
//   - error: 错误信息
func HasPermission(c *gin.Context, requiredPerm string) (bool, error) {
	claims := GetUserInfo(c)
	if claims == nil {
		return false, fmt.Errorf("用户未登录")
	}
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
	claims := GetUserInfo(c)
	if claims == nil {
		return false, fmt.Errorf("用户未登录")
	}
	return HasPermissionWithWildcardList(requiredPerm, claims.Authorities)
}

// HasPermissionWithList 使用权限列表检查权限
func HasPermissionWithList(requiredPerm string, userPerms []string) (bool, error) {
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
func HasPermissionWithWildcardList(requiredPerm string, userPerms []string) (bool, error) {
	if requiredPerm == "" {
		return false, nil
	}

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
func matchWildcard(pattern, str string) bool {
	if pattern == str {
		return true
	}
	if pattern == "*" {
		return true
	}
	return wildcardMatch(pattern, str, 0, 0)
}

// wildcardMatch 通配符匹配的动态规划实现
func wildcardMatch(pattern, str string, pIdx, sIdx int) bool {
	if pIdx == len(pattern) && sIdx == len(str) {
		return true
	}
	if pIdx == len(pattern) {
		return false
	}
	if pattern[pIdx] == '*' {
		return wildcardMatch(pattern, str, pIdx+1, sIdx) ||
			(sIdx < len(str) && wildcardMatch(pattern, str, pIdx, sIdx+1))
	}
	if sIdx < len(str) && (pattern[pIdx] == '?' || pattern[pIdx] == str[sIdx]) {
		return wildcardMatch(pattern, str, pIdx+1, sIdx+1)
	}
	return false
}

// BatchHasPermission 批量检查权限
// 参数：
//   - c: Gin上下文
//   - requiredPerms: 需要的权限列表
//   - requireAll: 是否需要满足所有权限（true:全部满足，false:任一满足）
func BatchHasPermission(c *gin.Context, requiredPerms []string, requireAll bool) (bool, error) {
	if len(requiredPerms) == 0 {
		return true, nil
	}

	claims := GetUserInfo(c)
	if claims == nil {
		return false, fmt.Errorf("用户未登录")
	}

	if requireAll {
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
	}

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

// HasAnyPermission 检查用户是否有任一权限
func HasAnyPermission(c *gin.Context, perms ...string) (bool, error) {
	return BatchHasPermission(c, perms, false)
}

// HasAllPermissions 检查用户是否有所有权限
func HasAllPermissions(c *gin.Context, perms ...string) (bool, error) {
	return BatchHasPermission(c, perms, true)
}
