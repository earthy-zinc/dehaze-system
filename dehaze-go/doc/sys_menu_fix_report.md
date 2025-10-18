# sys_menu 模块修复报告

## 修复时间

2025-10-18

## 修复概述

本次修复针对 sys_menu 模块中 Service 层逻辑一致性校验发现的所有 P0 高优先级问题进行了修复，确保 dehaze-go 实现与 dehaze-java 保持一致。

## 修复问题清单

### 1. 补充缺失的 ListRolePerms 方法

**问题描述**：

- dehaze-go 缺少 `ListRolePerms` 方法
- 该方法用于根据角色代码列表查询对应的权限集合
- Java 版本通过 3 表 INNER JOIN 实现

**修复方案**：
在 `dehaze-go/service/sys_menu.go` 中新增 `ListRolePerms` 方法：

```go
// ListRolePerms 获取角色权限集合
func (menuService *MenuService) ListRolePerms(roles []string) (perms []string, err error) {
 if len(roles) == 0 {
  return []string{}, nil
 }

 err = global.DB.Model(&model.SysMenu{}).
  Select("DISTINCT sys_menu.perm").
  Joins("INNER JOIN sys_role_menu ON sys_menu.id = sys_role_menu.menu_id").
  Joins("INNER JOIN sys_role ON sys_role.id = sys_role_menu.role_id").
  Where("sys_menu.type = ?", 4). // 按钮类型
  Where("sys_menu.perm IS NOT NULL AND sys_menu.perm != ?", "").
  Where("sys_role.code IN ?", roles).
  Pluck("sys_menu.perm", &perms).
  Error

 return perms, err
}
```

**对齐 Java 实现**：

```sql
SELECT DISTINCT t1.perm
FROM sys_menu t1
INNER JOIN sys_role_menu t2 ON t1.id = t2.menu_id
INNER JOIN sys_role t3 ON t3.id = t2.role_id
WHERE t1.type = 4 AND t1.perm IS NOT NULL
  AND t3.code IN (#{roles})
```

---

### 2. SaveMenu 方法缺少权限缓存刷新

**问题描述**：

- Java 版本在 `saveMenu` 方法中，当更新菜单（`menuForm.getId() != null`）后会调用 `roleMenuService.refreshRolePermsCache()` 刷新权限缓存
- dehaze-go 缺少此逻辑，可能导致菜单权限变更后缓存不一致

**修复方案**：
在 `SaveMenu` 方法中添加缓存刷新逻辑：

```go
// 更新菜单后清空所有角色权限缓存
if err == nil {
    menuService.clearAllRolePermsCache()
}
```

同时新增辅助方法：

```go
// clearAllRolePermsCache 清空所有角色权限缓存
func (menuService *MenuService) clearAllRolePermsCache() {
 if global.REDIS == nil {
  return
 }

 ctx := context.Background()
 // 删除整个role:perms哈希表
 global.REDIS.Del(ctx, "role:perms")
}
```

---

### 3. DeleteMenu 方法存在 SQL 注入风险

**问题描述**：

- Go 版本使用字符串拼接：`tree_path LIKE '%,' + id + '%'`
- 当 ID 为 1 时，会误匹配 ID=11, 21, 31 等（子串匹配问题）
- Java 版本使用 `CONCAT` 函数确保精确匹配

**修复方案**：
修改 SQL 查询，使用 CONCAT 函数避免子串误匹配：

```go
// DeleteMenu 删除菜单
func (menuService *MenuService) DeleteMenu(id int64) (err error) {
 // 删除菜单及其子菜单 - 修复SQL注入风险
 err = global.DB.Where("id = ? OR CONCAT(',',tree_path,',') LIKE CONCAT('%,',?,',%')", id, id).
  Delete(&model.SysMenu{}).
  Error

 // 删除成功后清空所有角色权限缓存
 if err == nil {
  menuService.clearAllRolePermsCache()
 }

 return err
}
```

同时删除了重复定义的旧版本方法（原第218-223行）。

**对齐 Java 实现**：

```java
boolean result = this.remove(new LambdaQueryWrapper<SysMenu>()
    .eq(SysMenu::getId, id)
    .or()
    .apply("CONCAT (',',tree_path,',') LIKE CONCAT('%,',{0},',%')", id));
```

---

### 4. DeleteMenu 方法重复定义

**问题描述**：

- 文件中存在两个 `DeleteMenu` 方法定义（第152行和第218行）
- 导致编译错误：`method MenuService.DeleteMenu already declared`

**修复方案**：
删除第218-223行的旧版本方法，仅保留修复后的版本（第152-169行）。

---

### 5. ListMenus 方法中存在未使用的 rootIds 变量

**问题描述**：

- 第36-52行定义并赋值了 `rootIds` 变量
- 但该变量在后续代码中未被使用
- 代码逻辑冗余

**修复方案**：
删除未使用的 rootIds 相关代码（第40-52行）：

```go
// 修复前
var rootIds []int64
for _, menu := range menus {
    isRoot := true
    for _, other := range menus {
        if menu.ID != other.ID && strings.Contains(menu.TreePath, ","+string(rune(other.ID))+",") {
            isRoot = false
            break
        }
    }
    if isRoot {
        rootIds = append(rootIds, menu.ID)
    }
}

// 修复后
// 直接构建菜单树，无需rootIds变量
menuList = buildMenuTree(0, menus)
```

---

### 6. ListRoutes 方法路由名转换不准确

**问题描述**：

- Go 版本使用 `strings.Title(strings.ReplaceAll(route.Path, "-", ""))`
- `strings.Title` 已在 Go 1.18 中废弃
- 转换结果不符合驼峰命名规范（例如："user-management" → "Usermanagement"，应为 "UserManagement"）

**Java 参考实现**：

```java
String routeName = StrUtil.toCamelCase(menu.getPath());
```

**修复方案**：

1. 在 `utils/type_convert.go` 中新增 `ToCamelCase` 函数：

```go
// ToCamelCase 将字符串转换为驼峰命名（首字母大写）
// 例如: "user-management" -> "UserManagement"
//       "hello-world" -> "HelloWorld"
func ToCamelCase(s string) string {
 if s == "" {
  return ""
 }

 // 分割字符串（按连字符、下划线、空格）
 words := strings.FieldsFunc(s, func(r rune) bool {
  return r == '-' || r == '_' || unicode.IsSpace(r)
 })

 // 将每个单词首字母大写
 for i, word := range words {
  if len(word) > 0 {
   runes := []rune(word)
   runes[0] = unicode.ToUpper(runes[0])
   words[i] = string(runes)
  }
 }

 return strings.Join(words, "")
}
```

2. 在 `service/sys_menu.go` 中引入 utils 包并使用新函数：

```go
import (
 // ...
 "github.com/earthyzinc/dehaze-go/utils"
 // ...
)

// buildRoutes 中修改路由名生成逻辑
routeVO := vo.RouteVO{
 Name:      utils.ToCamelCase(route.Path), // 路由 name 需要驼峰，首字母大写
 Path:      route.Path,
 Redirect:  route.Redirect,
 Component: route.Component,
 Meta:      meta,
}
```

---

## 修复后的文件清单

### 1. dehaze-go/service/sys_menu.go

- 新增 `ListRolePerms` 方法
- 新增 `clearAllRolePermsCache` 辅助方法
- 修复 `SaveMenu` 方法（添加缓存刷新）
- 修复 `DeleteMenu` 方法（SQL 注入 + 缓存刷新 + 删除重复定义）
- 修复 `ListMenus` 方法（删除未使用的 rootIds）
- 修复 `buildRoutes` 函数（使用 ToCamelCase）

### 2. dehaze-go/utils/type_convert.go

- 新增 `ToCamelCase` 函数

---

## 编译验证

执行编译命令验证修复结果：

```bash
cd dehaze-go && go build ./service
```

**结果**：✅ 编译通过，无错误

---

## 待处理事项（P1优先级）

根据逻辑一致性报告，以下问题暂未修复（影响较小）：

1. **SaveMenu 外链处理不一致**
   - Java: `menuForm.setComponent(null)`
   - Go: `menuForm.Component = ""`
   - 建议：统一为空字符串或 NULL，需确认前端是否依赖此行为

---

## 下一步计划

1. ✅ **第一步：功能完整性比对** - 已完成
2. ✅ **第二步：Service层逻辑一致性校验** - 已完成
3. ✅ **P0问题修复** - 已完成
4. ⏳ **第三步：代码质量与正确性检查** - 待进行
5. ⏳ **第四步：测试用例完善与验证** - 待进行

---

## 总结

本次修复共处理 **6 个 P0 高优先级问题**：

- ✅ 补充缺失方法（ListRolePerms）
- ✅ 权限缓存刷新机制（SaveMenu + DeleteMenu）
- ✅ SQL 注入风险修复（DeleteMenu）
- ✅ 编译错误修复（重复方法定义）
- ✅ 代码冗余清理（未使用变量）
- ✅ 路由名转换优化（ToCamelCase）

所有修复均已对齐 dehaze-java 实现逻辑，编译通过，可进入下一阶段审查。
