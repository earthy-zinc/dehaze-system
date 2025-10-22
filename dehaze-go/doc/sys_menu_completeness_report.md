# sys_menu 模块功能完整性比对报告

## 一、API 接口对比

### Java Controller (SysMenuController)

✅ **完整性：100%**

| 接口方法 | 路由 | Java实现 | Go实现 | 状态 |
|---------|------|---------|--------|------|
| listMenus | GET /api/v1/menus | ✅ | ✅ | 一致 |
| listMenuOptions | GET /api/v1/menus/options | ✅ | ✅ | 一致 |
| listRoutes | GET /api/v1/menus/routes | ✅ | ✅ | 一致 |
| getMenuForm | GET /api/v1/menus/{id}/form | ✅ | ✅ | 一致 |
| addMenu | POST /api/v1/menus | ✅ | ✅ | 一致 |
| updateMenu | PUT /api/v1/menus/{id} | ✅ | ✅ | 一致 |
| deleteMenu | DELETE /api/v1/menus/{id} | ✅ | ✅ | 一致 |
| updateMenuVisible | PATCH /api/v1/menus/{menuId} | ✅ | ✅ | 一致 |

**结论：** Go 实现已完整覆盖所有 Java Controller 接口。

---

## 二、Service 层业务方法对比

### Java Service Interface (SysMenuService)

✅ **完整性：87.5%**

| 方法名 | Java实现 | Go实现 | 状态 |
|-------|---------|--------|------|
| listMenus | ✅ | ✅ | 一致 |
| listMenuOptions | ✅ | ✅ | 一致 |
| listRoutes | ✅ | ✅ | 一致 |
| saveMenu | ✅ | ✅ | 一致 |
| updateMenuVisible | ✅ | ✅ | 一致 |
| getMenuForm | ✅ | ✅ | 一致 |
| deleteMenu | ✅ | ✅ | 一致 |
| **listRolePerms** | ✅ | ❌ | **缺失** |

**发现问题：**

1. ❌ **缺失方法：listRolePerms** - 获取角色权限集合
   - Java定义：`Set<String> listRolePerms(Set<String> roles)`
   - 用途：根据角色代码集合获取对应的权限标识集合
   - Go实现：完全缺失此方法

---

## 三、Mapper/SQL 对比

### Java Mapper (SysMenuMapper.xml)

✅ **SQL覆盖度：50%**

| SQL方法 | Java实现 | Go实现 | 状态 |
|---------|---------|--------|------|
| listRoutes | ✅ 复杂SQL（3表LEFT JOIN） | ✅ GORM实现 | 一致 |
| **listRolePerms** | ✅ 复杂SQL（3表INNER JOIN） | ❌ | **缺失** |

#### Java listRoutes SQL

```sql
SELECT t1.id, t1.name, t1.parent_id, t1.path, t1.component,
       t1.icon, t1.sort, t1.visible, t1.redirect, t1.type,
       t3.code, t1.always_show, t1.keep_alive
FROM sys_menu t1
LEFT JOIN sys_role_menu t2 ON t1.id = t2.menu_id
LEFT JOIN sys_role t3 ON t2.role_id = t3.id
WHERE t1.type != 4
ORDER BY t1.sort ASC
```

#### Go listRoutes 实现

```go
global.DB.Model(&model.SysMenu{}).
    Select("...").
    Joins("LEFT JOIN sys_role_menu ON sys_menu.id = sys_role_menu.menu_id").
    Joins("LEFT JOIN sys_role ON sys_role_menu.role_id = sys_role.id").
    Where("sys_menu.type != ?", 4).
    Order("sys_menu.sort ASC").
    Find(&routeBOs)
```

**问题：** Go 实现缺少 `listRolePerms` SQL查询。

---

## 四、数据模型对比

### Java Entity vs Go Model

✅ **字段完整性：100%**

| 字段名 | Java类型 | Go类型 | Java注释 | Go注释 | 状态 |
|-------|---------|--------|---------|--------|------|
| id | Long | int64 | 菜单ID | ✅ | 一致 |
| parentId | Long | int64 | 父菜单ID | 父菜单ID | 一致 |
| name | String | string | 菜单名称 | 菜单名称 | 一致 |
| type | MenuTypeEnum | int8 | 菜单类型(1-菜单；2-目录；3-外链；4-按钮权限) | 菜单类型(1:菜单 2:目录 3:外链 4:按钮) | 一致 |
| path | String | string | 路由路径(浏览器地址栏路径) | 路由路径(浏览器地址栏路径) | 一致 |
| component | String | string | 组件路径(vue页面完整路径，省略.vue后缀) | 组件路径(vue页面完整路径，省略.vue后缀) | 一致 |
| perm | String | string | 权限标识 | 权限标识 | 一致 |
| visible | Integer | int8 | 显示状态(1:显示;0:隐藏) | 显示状态(1-显示;0-隐藏) | 一致 |
| sort | Integer | int | 排序 | 排序 | 一致 |
| icon | String | string | 菜单图标 | 菜单图标 | 一致 |
| redirect | String | string | 跳转路径 | 跳转路径 | 一致 |
| treePath | String | string | 父节点路径，以英文逗号(,)分割 | 父节点ID路径 | 一致 |
| keepAlive | Integer | int8 | 【菜单】是否开启页面缓存(1:开启;0:关闭) | 【菜单】是否开启页面缓存(1:是 0:否) | 一致 |
| alwaysShow | Integer | int8 | 【目录】只有一个子路由是否始终显示(1:是 0:否) | 【目录】只有一个子路由是否始终显示(1:是 0:否) | 一致 |

**结论：** 所有字段已完整映射，类型对应关系正确。

---

## 五、功能缺失汇总

### ❌ 缺失功能清单

#### 1. **缺失方法：listRolePerms**

- **位置：** service/sys_menu.go
- **Java签名：** `Set<String> listRolePerms(Set<String> roles)`
- **功能描述：** 根据角色代码集合查询对应的权限标识集合
- **SQL逻辑：**

  ```sql
  SELECT DISTINCT t1.perm
  FROM sys_menu t1
  INNER JOIN sys_role_menu t2 ON t1.id = t2.menu_id
  INNER JOIN sys_role t3 ON t3.id = t2.role_id
  WHERE t1.type = 4  -- 按钮类型
    AND t1.perm IS NOT NULL
    AND t3.code IN (#{roles})
  ```

- **影响范围：** 权限验证功能可能受影响
- **优先级：** 🔴 **高** - 权限系统核心功能

---

## 六、第一步总结

### ✅ 已实现功能（87.5%）

1. ✅ 菜单列表查询（含树形结构构建）
2. ✅ 菜单下拉选项（含递归树构建）
3. ✅ 路由列表查询（含多表JOIN）
4. ✅ 菜单表单数据获取
5. ✅ 菜单新增/修改（含树路径生成）
6. ✅ 菜单显示状态修改
7. ✅ 菜单删除（含级联删除子菜单）

### ❌ 待补充功能（12.5%）

1. ❌ **listRolePerms** - 获取角色权限集合

### 🔍 细节问题

#### 1. ListMenus 方法中的 rootIds 计算逻辑问题

**问题代码：**

```go
// Go实现 - 第36-45行
var rootIds []int64
for _, menu := range menus {
    isRoot := true
    for _, other := range menus {
        // 检查当前菜单是否是其他菜单的子菜单
        if menu.ID != other.ID && strings.Contains(menu.TreePath, ","+string(rune(other.ID))+",") {
            isRoot = false
            break
        }
    }
    if isRoot {
        rootIds = append(rootIds, menu.ID)
    }
}
```

**问题：**

- 计算了 `rootIds` 但从未使用
- `buildMenuTree(0, menus)` 直接传入 parentId=0

**Java实现：**

```java
List<Long> rootIds = TreeDataUtils.findRootIds(menus, SysMenu::getId, SysMenu::getParentId);
return rootIds.stream()
    .flatMap(rootId -> buildMenuTree(rootId, menus).stream())
    .toList();
```

**建议：** 删除未使用的 rootIds 计算逻辑，保持与当前实现一致。

---

## 七、待审核事项

📋 **第一步完成，等待用户审核确认：**

1. ✅ 确认功能完整性比对结果
2. ❌ **需补充 `listRolePerms` 方法**（优先级：高）
3. ⚠️ 确认 `ListMenus` 中未使用的 `rootIds` 是否需要清理
4. ✅ 所有字段映射正确

**下一步：** 用户审核通过后，进入第二步 - Service层逻辑一致性校验
