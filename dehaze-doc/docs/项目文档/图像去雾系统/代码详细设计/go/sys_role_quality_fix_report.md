# sys_role 模块 - 高优先级问题修复报告

## 修复概述

**修复时间**: 2025-10-18  
**修复内容**: 第三步代码质量检查中发现的 3 个高优先级问题  
**修复文件**: `dehaze-go/service/sys_role.go`

---

## 修复详情

### 1. 全局变量空指针风险修复 ✅

**问题描述**: 所有方法直接使用 `global.DB` 和 `global.LOG` 而未进行空指针检查

**修复方案**:

- 在所有使用 `global.DB` 的方法开头添加空指针检查
- 在 `loadRolePermsToCache` 方法中添加 `global.LOG` 空指针检查

**修复代码示例**:

```go
// 修复前
func (roleService *RoleService) GetRolePage(queryParams query.RolePageQuery) (result vo.PageResult[vo.RolePageVO], err error) {
    // 直接使用 global.DB
    db := global.DB.Model(&model.SysRole{})
    ...
}

// 修复后
func (roleService *RoleService) GetRolePage(queryParams query.RolePageQuery) (result vo.PageResult[vo.RolePageVO], err error) {
    // 检查全局数据库连接
    if global.DB == nil {
        return result, errors.New("数据库连接未初始化")
    }
    
    db := global.DB.Model(&model.SysRole{})
    ...
}
```

**受影响的方法** (共 9 个):

1. `GetRolePage` - 添加 DB 空指针检查
2. `ListRoleOptions` - 添加 DB 空指针检查
3. `SaveRole` - 添加 DB 空指针检查
4. `GetRoleForm` - 添加 DB 空指针检查
5. `UpdateRoleStatus` - 添加 DB 空指针检查
6. `DeleteRoles` - 添加 DB 空指针检查
7. `GetRoleMenuIds` - 添加 DB 空指针检查
8. `AssignMenusToRole` - 添加 DB 空指针检查
9. `GetMaximumDataScope` - 添加 DB 空指针检查
10. `loadRolePermsToCache` - 添加 DB 和 LOG 空指针检查

---

### 2. 输入参数验证缺失修复 ✅

**问题描述**: `SaveRole` 和 `UpdateRoleStatus` 方法缺少对输入参数的验证

**修复方案**:

- 在 `SaveRole` 中添加角色名称、编码、状态的完整验证
- 在 `UpdateRoleStatus` 中添加状态值的验证

**修复代码**:

#### SaveRole 方法参数验证

```go
// 输入参数验证
if strings.TrimSpace(roleFormBO.Name) == "" {
    return errors.New("角色名称不能为空")
}
if len(roleFormBO.Name) > 50 {
    return errors.New("角色名称长度不能超过50个字符")
}
if strings.TrimSpace(roleFormBO.Code) == "" {
    return errors.New("角色编码不能为空")
}
if len(roleFormBO.Code) > 50 {
    return errors.New("角色编码长度不能超过50个字符")
}
if roleFormBO.Status != 0 && roleFormBO.Status != 1 {
    return errors.New("角色状态值无效，必须为0或1")
}
```

#### UpdateRoleStatus 方法参数验证

```go
// 输入参数验证
if status != 0 && status != 1 {
    return errors.New("角色状态值无效，必须为0或1")
}
```

**验证规则**:

- 角色名称: 不能为空、不能为纯空格、长度不超过50字符
- 角色编码: 不能为空、不能为纯空格、长度不超过50字符
- 角色状态: 必须为0(禁用)或1(启用)

---

### 3. N+1 查询问题优化 ✅

**问题描述**: `DeleteRoles` 方法在循环中逐个查询每个角色的用户关联数量，造成 N+1 查询性能问题

**修复方案**: 使用批量查询优化，一次性查询所有角色的用户关联情况

**修复前代码** (存在 N+1 问题):

```go
// 检查角色是否被用户关联并逐个删除
for _, role := range roles {
    var userRoleCount int64
    err = tx.Model(&model.SysUserRole{}).
        Where("role_id = ?", role.ID).
        Count(&userRoleCount).Error  // 每个角色执行一次查询
    
    if userRoleCount > 0 {
        return errors.New("角色【" + role.Name + "】已分配用户")
    }
    
    // 逐个删除角色...
}
```

**修复后代码** (批量查询):

```go
// 批量检查角色是否被用户关联（优化N+1查询）
roleIds := make([]int64, len(roles))
roleMap := make(map[int64]*model.SysRole)
for i, role := range roles {
    roleIds[i] = role.ID
    roleMap[role.ID] = &roles[i]
}

// 一次性查询所有角色的用户关联数量
type RoleUserCount struct {
    RoleID int64
    Count  int64
}
var roleUserCounts []RoleUserCount
err = tx.Model(&model.SysUserRole{}).
    Select("role_id, COUNT(*) as count").
    Where("role_id IN ?", roleIds).
    Group("role_id").
    Find(&roleUserCounts).Error

// 检查是否有角色被用户关联
for _, ruc := range roleUserCounts {
    if ruc.Count > 0 {
        role := roleMap[ruc.RoleID]
        return errors.New("角色【" + role.Name + "】已分配用户")
    }
}

// 批量逻辑删除角色
err = tx.Model(&model.SysRole{}).
    Where("id IN ?", roleIds).
    Updates(map[string]interface{}{
        "deleted":     1,
        "update_time": time.Now(),
    }).Error
```

**优化效果**:

- **修复前**: N 个角色需要执行 N+1 次数据库查询 (1次查角色 + N次查关联)
- **修复后**: 仅需执行 3 次数据库查询 (1次查角色 + 1次批量查关联 + 1次批量删除)
- **性能提升**: 当删除 10 个角色时，查询次数从 11 次降至 3 次，提升约 73%

**额外优化**:

- 将权限缓存刷新移至事务提交之后，确保数据一致性
- 使用批量删除替代循环删除，进一步提升性能

---

## 修复验证

### 代码编译验证

```bash
cd dehaze-go
go build ./service/sys_role.go
```

预期: 无编译错误

### 单元测试验证

```bash
go test -v ./test/sys_role_test.go -run TestRoleService
```

预期: 所有测试用例通过

---

## 修复总结

### 已完成的修复 ✅

1. ✅ 全局变量空指针风险 - 所有方法添加 DB/LOG 空指针检查
2. ✅ 输入参数验证缺失 - SaveRole 和 UpdateRoleStatus 添加完整验证
3. ✅ N+1 查询性能问题 - DeleteRoles 优化为批量查询

### 修复影响

- **安全性**: 消除空指针崩溃风险，提升系统稳定性
- **数据完整性**: 严格的输入验证防止无效数据进入数据库
- **性能**: DeleteRoles 方法查询次数大幅减少，响应时间显著降低

### 后续建议

1. 进入第四步：测试用例完善与验证
2. 编写针对修复问题的专项测试用例
3. 进行压力测试验证性能优化效果

---

**修复完成状态**: ✅ 所有高优先级问题已修复  
**下一步**: 第四步 - 测试用例完善与验证
