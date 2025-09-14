package initialize

import (
	"context"
	"strings"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/global"
)

// 初始化角色权限缓存
func InitRolePermsCache() error {
	// 清理旧缓存
	if err := ClearRolePermsCache(); err != nil {
		return err
	}

	// 查询数据库
	rows, err := global.DB.Raw(`
        SELECT 
            t2.code AS role_code,
            t3.perm
        FROM 
            sys_role_menu t1
        INNER JOIN sys_role t2 ON t1.role_id = t2.id AND t2.deleted = 0 AND t2.status = 1
        INNER JOIN sys_menu t3 ON t1.menu_id = t3.id
        WHERE 
            t3.type = 4
        ORDER BY t2.code
    `).Rows()

	if err != nil {
		return err
	}
	defer rows.Close()

	// 按角色分组权限
	rolePermsMap := make(map[string][]string)
	for rows.Next() {
		var roleCode string
		var perm string
		if err := global.DB.ScanRows(rows, &struct {
			RoleCode *string `gorm:"column:role_code"`
			Perm     *string `gorm:"column:perm"`
		}{
			RoleCode: &roleCode,
			Perm:     &perm,
		}); err != nil {
			return err
		}

		rolePermsMap[roleCode] = append(rolePermsMap[roleCode], perm)
	}

	// 缓存到本地和Redis
	for roleCode, perms := range rolePermsMap {
		// 写入Redis
		if global.REDIS != nil {
			_, err := global.REDIS.HSet(context.Background(), "role_perms", roleCode, strings.Join(perms, ",")).Result()
			if err != nil {
				return err
			}
		} else {
			// 写入本地缓存
			// 注意：根据实际使用情况判断是否需要本地缓存，对于频繁访问的权限数据，本地缓存可以提高性能
			// 但在分布式环境中，本地缓存可能导致数据不一致问题
			// 这里保留本地缓存以提高访问速度，但需要在更新权限时同步清理所有节点的本地缓存
			global.LOCAL_CACHE.SetDefault(common.ROLE_PERMS_PREFIX+roleCode, perms)
		}
	}

	return nil
}

// 清理旧权限缓存
func ClearRolePermsCache() error {
	// 清理本地缓存
	// 注意：local_cache.Cache没有提供直接的模式匹配删除方法，需要遍历所有key
	// 在实际应用中，如果本地缓存key较多，可能需要考虑其他策略

	// 清理Redis缓存
	if global.REDIS != nil {
		// 删除整个hash
		_, err := global.REDIS.Del(context.Background(), "role_perms").Result()
		if err != nil {
			return err
		}
	}

	return nil
}

// 角色权限BO
type RolePermsBO struct {
	RoleCode string `gorm:"column:role_code"`
	Perm     string `gorm:"column:perm"`
}
