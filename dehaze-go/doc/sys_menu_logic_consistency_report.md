# sys_menu 模块 Service 层逻辑一致性校验报告

## 一、方法逻辑对比分析

### 1. ListMenus 方法

#### Java 实现逻辑

```java
List<SysMenu> menus = this.list(new LambdaQueryWrapper<SysMenu>()
    .like(CharSequenceUtil.isNotBlank(queryParams.getKeywords()), SysMenu::getName, queryParams.getKeywords())
    .orderByAsc(SysMenu::getSort)
);
List<Long> rootIds = TreeDataUtils.findRootIds(menus, SysMenu::getId, SysMenu::getParentId);
return rootIds.stream()
    .flatMap(rootId -> buildMenuTree(rootId, menus).stream())
    .toList();
```

#### Go 实现逻辑

```go
db := global.DB.Model(&model.SysMenu{})
if queryParams.Keywords != "" {
    keyword := "%" + queryParams.Keywords + "%"
    db = db.Where("name LIKE ?", keyword)
}
if queryParams.Status != nil {
    db = db.Where("visible = ?", *queryParams.Status)
}
err = db.Order("sort ASC").Find(&menus).Error

// 未使用的 rootIds 计算
var rootIds []int64
for _, menu := range menus {...}

menuList = buildMenuTree(0, menus)
return menuList, nil
```

#### ⚠️ 逻辑差异

1. **Java**: 使用 `TreeDataUtils.findRootIds` 动态找根节点，支持多根树
2. **Go**: 直接传 `parentId=0`，假设所有根节点的 `parentId=0`
3. **Go**: 计算了 `rootIds` 但未使用（冗余代码）
4. **Java**: 不支持 status 过滤
5. **Go**: 额外支持 `status` (visible) 过滤条件

**一致性评分**: 75% - 核心逻辑基本一致，但实现方式有差异

---

### 2. ListMenuOptions 方法

#### Java 实现逻辑

```java
List<SysMenu> menuList = this.list(new LambdaQueryWrapper<SysMenu>()
    .orderByAsc(SysMenu::getSort));
return buildMenuOptions(SystemConstants.ROOT_NODE_ID, menuList);
```

#### Go 实现逻辑

```go
var menuList []model.SysMenu
err = global.DB.Model(&model.SysMenu{}).
    Order("sort ASC").
    Find(&menuList).Error
options = buildMenuOptions(0, menuList)
```

#### ✅ 逻辑一致性

- 查询条件：完全一致（无条件查询 + 按 sort 升序）
- 构建逻辑：完全一致（递归构建树形结构）
- ROOT_NODE_ID 在两边都是 0

**一致性评分**: 100%

---

### 3. ListRoutes 方法

#### Java SQL 实现

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

#### Go GORM 实现

```go
global.DB.Model(&model.SysMenu{}).
    Select("sys_menu.id, sys_menu.parent_id, sys_menu.name, sys_menu.path, sys_menu.component, sys_menu.icon, sys_menu.sort, sys_menu.visible, sys_menu.redirect, sys_menu.type, sys_menu.always_show, sys_menu.keep_alive, sys_role.code").
    Joins("LEFT JOIN sys_role_menu ON sys_menu.id = sys_role_menu.menu_id").
    Joins("LEFT JOIN sys_role ON sys_role_menu.role_id = sys_role.id").
    Where("sys_menu.type != ?", 4).
    Order("sys_menu.sort ASC").
    Find(&routeBOs)
```

#### ✅ 逻辑一致性

- SQL 查询：完全一致（3表 LEFT JOIN）
- WHERE 条件：完全一致（排除按钮类型）
- 字段选择：完全一致
- 排序规则：完全一致

**Java Route 转换逻辑**:

```java
String routeName = StringUtils.capitalize(CharSequenceUtil.toCamelCase(routeBO.getPath(), '-'));
routeVO.setName(routeName);
```

**Go Route 转换逻辑**:

```go
Name: strings.Title(strings.ReplaceAll(route.Path, "-", ""))
```

#### ⚠️ 细微差异

- **Java**: 使用 Hutool 的 `toCamelCase` 转驼峰 + `capitalize` 首字母大写
- **Go**: 简单替换 `-` + `strings.Title`（已废弃函数）
- **示例**:
    - 路径 `user-management`
    - Java: `UserManagement`
    - Go: `Usermanagement` ❌

**一致性评分**: 95% - SQL 完全一致，路由名转换有差异

---

### 4. SaveMenu 方法

#### Java 实现逻辑

```java
MenuTypeEnum menuType = menuForm.getType();

if (menuType == MenuTypeEnum.CATALOG) {  // 目录
    String path = menuForm.getPath();
    if (menuForm.getParentId() == 0 && !path.startsWith("/")) {
        menuForm.setPath("/" + path);
    }
    menuForm.setComponent("Layout");
} else if (menuType == MenuTypeEnum.EXTLINK) {  // 外链
    menuForm.setComponent(null);
}

SysMenu entity = menuConverter.form2Entity(menuForm);
String treePath = generateMenuTreePath(menuForm.getParentId());
entity.setTreePath(treePath);

boolean result = this.saveOrUpdate(entity);
if (result && menuForm.getId() != null) {
    roleMenuService.refreshRolePermsCache();  // 刷新缓存
}
```

#### Go 实现逻辑

```go
menuType := menuForm.Type

if menuType == 2 { // 目录
    path := menuForm.Path
    if menuForm.ParentID == 0 && !strings.HasPrefix(path, "/") {
        menuForm.Path = "/" + path
    }
    menuForm.Component = "Layout"
} else if menuType == 3 { // 外链
    menuForm.Component = ""
}

treePath := menuService.generateMenuTreePath(menuForm.ParentID)

menu := model.SysMenu{...}  // 手动构建

if menuForm.ID != nil {
    menu.ID = *menuForm.ID
    menu.BaseModel.UpdatedAt = time.Now()
    err = global.DB.Save(&menu).Error
} else {
    err = global.DB.Create(&menu).Error
}
```

#### ❌ 关键缺失

1. **缺少权限缓存刷新**: Go 实现没有调用 `refreshRolePermsCache()`
2. **类型常量**: Java 使用枚举 `MenuTypeEnum.CATALOG(2)`, Go 硬编码数字 `2`
3. **外链处理**: Java 设置 `null`, Go 设置空字符串 `""`

**一致性评分**: 80% - 核心逻辑一致，但缺少缓存刷新

---

### 5. UpdateMenuVisible 方法

#### Java 实现

```java
return this.update(new LambdaUpdateWrapper<SysMenu>()
    .eq(SysMenu::getId, menuId)
    .set(SysMenu::getVisible, visible)
);
```

#### Go 实现

```go
err = global.DB.Model(&model.SysMenu{}).
    Where("id = ?", menuId).
    Update("visible", visible).
    Error
```

#### ✅ 逻辑一致性

- 更新逻辑：完全一致
- 无额外校验
- 无缓存操作

**一致性评分**: 100%

---

### 6. GetMenuForm 方法

#### Java 实现

```java
SysMenu entity = this.getById(id);
return menuConverter.entity2Form(entity);
```

#### Go 实现

```go
var entity model.SysMenu
err = global.DB.Where("id = ?", id).First(&entity).Error
if err != nil {
    if errors.Is(err, gorm.ErrRecordNotFound) {
        return menuForm, errors.New("菜单不存在")
    }
    return menuForm, err
}

idPtr := entity.ID
menuForm = bo.MenuForm{
    ID: &idPtr,
    ParentID: entity.ParentID,
    ...  // 手动映射字段
}
```

#### ⚠️ 差异

- **Java**: 使用 MapStruct 自动转换
- **Go**: 手动字段映射
- **Go**: 额外的错误处理（记录不存在）

**一致性评分**: 95% - 逻辑一致，实现方式不同

---

### 7. DeleteMenu 方法

#### Java 实现

```java
boolean result = this.remove(new LambdaQueryWrapper<SysMenu>()
    .eq(SysMenu::getId, id)
    .or()
    .apply("CONCAT (',',tree_path,',') LIKE CONCAT('%,',{0},',%')", id));

if (result) {
    roleMenuService.refreshRolePermsCache();  // 刷新缓存
}
return result;
```

#### Go 实现

```go
err = global.DB.Where("id = ? OR tree_path LIKE ?", id, "%,"+strconv.FormatInt(id, 10)+"%").
    Delete(&model.SysMenu{}).
    Error
return err
```

#### ❌ 关键缺失

1. **缺少权限缓存刷新**: Go 实现没有刷新缓存
2. **SQL 略有差异**:
    - Java: `CONCAT (',',tree_path,',') LIKE CONCAT('%,',{0},',%')`
    - Go: `tree_path LIKE '%,{id},%'`
    - Go 的实现可能匹配到不完整的 ID（如 tree_path="0,12" 会匹配 id=1）

**一致性评分**: 70% - 核心逻辑相似，但缺少缓存且 SQL 有风险

---

### 8. ListRolePerms 方法（新增）

#### Java 实现

```sql
SELECT DISTINCT t1.perm
FROM sys_menu t1
INNER JOIN sys_role_menu t2 ON t1.id = t2.menu_id
INNER JOIN sys_role t3 ON t3.id = t2.role_id
WHERE t1.type = 4
  AND t1.perm IS NOT NULL
  AND t3.code IN (#{roles})
```

#### Go 实现

```go
err = global.DB.Model(&model.SysMenu{}).
    Select("DISTINCT sys_menu.perm").
    Joins("INNER JOIN sys_role_menu ON sys_menu.id = sys_role_menu.menu_id").
    Joins("INNER JOIN sys_role ON sys_role.id = sys_role_menu.role_id").
    Where("sys_menu.type = ?", 4).
    Where("sys_menu.perm IS NOT NULL AND sys_menu.perm != ?", "").
    Where("sys_role.code IN ?", roles).
    Pluck("sys_menu.perm", &perms).
    Error
```

#### ✅ 逻辑一致性

- SQL 结构：完全一致（3表 INNER JOIN）
- WHERE 条件：完全一致
- 空值处理：Go 额外过滤空字符串
- 空角色处理：Go 返回空数组（Java 返回空 Set）

**一致性评分**: 100%

---

## 二、发现的问题汇总

### 🔴 高优先级问题

#### 1. 缺少权限缓存刷新机制

- **影响方法**: `SaveMenu`, `DeleteMenu`
- **Java 行为**: 修改/删除菜单后调用 `roleMenuService.refreshRolePermsCache()`
- **Go 缺失**: 完全没有缓存刷新逻辑
- **影响**: 菜单权限变更后，用户权限不会实时更新

#### 2. DeleteMenu SQL 注入风险

- **当前 SQL**: `tree_path LIKE '%,{id},%'`
- **问题**: 可能匹配到 ID 的子串（如 id=1 会匹配 tree_path="0,12"）
- **Java 方案**: `CONCAT (',',tree_path,',') LIKE CONCAT('%,',{0},',%')`
- **建议**: 修改为 `CONCAT(',',tree_path,',') LIKE CONCAT('%,', ?, ',%')`

### ⚠️ 中优先级问题

#### 3. ListMenus 未使用的 rootIds 变量

- **位置**: service/sys_menu.go 第 36-52 行
- **问题**: 计算了 rootIds 但从未使用
- **建议**: 删除冗余代码或参考 Java 实现使用动态根节点

#### 4. ListRoutes 路由名转换不准确

- **当前实现**: `strings.Title(strings.ReplaceAll(route.Path, "-", ""))`
- **问题**:
    - `strings.Title` 已废弃（Go 1.18+）
    - 转换结果不符合驼峰命名（`user-management` → `Usermanagement`）
- **Java 实现**: 使用 Hutool 的 `toCamelCase` 转驼峰
- **建议**: 实现正确的驼峰转换函数

#### 5. SaveMenu 外链处理不一致

- **Java**: 设置 `component = null`
- **Go**: 设置 `component = ""`
- **影响**: 数据库存储值不一致（NULL vs 空字符串）

### ℹ️ 低优先级问题

#### 6. 缺少输入校验

- Java 在 Controller 层有 `@Valid` 注解校验
- Go 当前无任何输入校验（如菜单名称长度、类型范围等）

---

## 三、逻辑一致性评分

| 方法                | 一致性评分 | 主要问题               |
|-------------------|-------|--------------------|
| ListMenus         | 75%   | 未使用 rootIds，实现方式差异 |
| ListMenuOptions   | 100%  | ✅ 完全一致             |
| ListRoutes        | 95%   | 路由名转换有差异           |
| SaveMenu          | 80%   | 缺少缓存刷新             |
| UpdateMenuVisible | 100%  | ✅ 完全一致             |
| GetMenuForm       | 95%   | 实现方式不同             |
| DeleteMenu        | 70%   | 缺少缓存刷新 + SQL 风险    |
| ListRolePerms     | 100%  | ✅ 完全一致             |

**总体一致性**: **89%**

---

## 四、修复建议优先级

### 立即修复（P0）

1. ✅ 补充 `ListRolePerms` 方法（已完成）
2. ❌ 在 `SaveMenu` 和 `DeleteMenu` 中添加权限缓存刷新
3. ❌ 修复 `DeleteMenu` 的 SQL 注入风险

### 近期修复（P1）

4. ❌ 删除 `ListMenus` 中未使用的 rootIds 代码
5. ❌ 修复 `ListRoutes` 的路由名转换逻辑
6. ❌ 统一 `SaveMenu` 外链的 component 处理

### 后续优化（P2）

7. ❌ 添加输入参数校验
8. ❌ 统一错误处理机制

---

## 五、下一步操作

等待用户审核确认后，执行以下修复：

1. 实现权限缓存刷新机制
2. 修复 DeleteMenu SQL
3. 优化代码质量
4. 进入第三步：代码质量与正确性检查
