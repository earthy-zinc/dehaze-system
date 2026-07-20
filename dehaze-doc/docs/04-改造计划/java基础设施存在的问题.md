# dehaze-java 后端特定问题与改进建议

> 文档定位：仅收录 **Java 特有问题**（性能/优化/bug），即由 Java 语言特性或 Spring 框架特性导致的实现缺陷。
>
> 核对基准日期：2026-07-21
> 审查范围：`dehaze-java` 全部基础设施层代码
> 审查方法：对照 [09-后端基础设施设计(Java)](../02-系统架构/09-后端基础设施设计(Java).md) 核对实际实现

---

## 一、问题总览

| 领域 | 严重 | 高 | 中 | 低 |
|------|------|---|---|---|
| 安全认证 | 1 | 0 | 0 | 0 |
| 事务与数据一致性 | 0 | 1 | 0 | 0 |
| 外部服务客户端 | 0 | 0 | 1 | 0 |
| 数据访问 | 0 | 0 | 1 | 0 |

---

## 二、安全认证（Java/Spring Security 特有）

### 2.1 [SEVERE] 账户锁定与过期机制完全缺失

[SysUserDetails.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/security/model/SysUserDetails.java) 未重写 `isAccountNonLocked`/`isAccountNonExpired`/`isCredentialsNonExpired`，全部使用 `UserDetails` 接口默认实现（恒返回 `true`）。[SysUser.java](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/model/entity/SysUser.java) 实体无 `failed_attempts`/`lock_until` 字段。全库无 `AuthenticationFailureHandler`/`AuthenticationSuccessHandler` 实现。

**影响**：即使密码被暴力破解成功，账户也不会被锁定；密码过期策略无法实施。虽然登录接口已通过 `@RateLimit` 限制单 IP 60 秒内 10 次（[AuthController.java:27-28](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/controller/AuthController.java#L27)），但分布式攻击下仍可能成功。

**改进建议**：
- `SysUser` 增加 `failed_attempts`（int）、`lock_until`（datetime）字段
- `SysUserDetailsService` 实现 `isAccountNonLocked`：检查 `lock_until > now()`
- `AuthenticationFailureHandler` / `AuthenticationSuccessHandler` 中更新失败计数

### 2.2 [SEVERE] 开发环境 JWT 密钥硬编码 fallback

[application-dev.yml:110](file:///e:/DehazeSystem/dehaze-java/src/main/resources/application-dev.yml#L110)：

```yaml
key: ${JWT_SECRET_KEY:SecretKey012345678901234567890123456789012345678901234567890123456789}
```

未设置环境变量时回退到公开的硬编码密钥。生产环境（`application-prod.yml:74`）已修复（`key: ${JWT_SECRET_KEY}` 无 fallback），但开发环境若被外部访问，攻击者可伪造任意用户 JWT。

**改进建议**：开发环境也移除 fallback，强制要求设置 `JWT_SECRET_KEY` 环境变量（或在 IDE 启动配置中显式指定）。

---

## 三、事务与数据一致性（Spring @Transactional 特有）

### 3.1 [HIGH] 长事务仍包裹 MinIO 外部调用（2 处）

以下 2 个方法仍是 `@Transactional`，内部调用 MinIO 删除，会导致数据库连接被长时间占用、高并发下连接池耗尽：

| 方法 | 文件:行 | MinIO 调用场景 |
|------|---------|---------------|
| `deleteDatasetItemCascade` | [DatasetOperationServiceImpl.java:312](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/DatasetOperationServiceImpl.java#L312) | 委托 `batchDeleteDatasetItemsCascadeWithResult` → 循环 `deleteFile`，每次 2 次 MinIO `removeObject` |
| `SysItemFileServiceImpl.deleteFile` | [SysItemFileServiceImpl.java:126](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/SysItemFileServiceImpl.java#L126) | 直接调用 `sysFileService.deleteFile` → `minioClient.removeObject`（line 139、144 共 2 次） |

> 注：`createDatasetItemWithImages` 已修复——公共方法和私有 `doCreateDatasetItemWithImages` 均移除 `@Transactional`，MinIO 上传在事务外执行，DB 写入通过 `self.saveItemFileRecord` 走短事务（[SysItemFileServiceImpl.java:65-95](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/SysItemFileServiceImpl.java#L65)）。

**影响**：MinIO 慢响应或网络抖动时，数据库连接被占用数秒到数十秒，高并发下连接池迅速耗尽。

**改进建议**：采用"先 DB 标记 + 事务提交后批量清理 MinIO"的补偿模式——事务内仅记录待删除文件清单到内存或 Redis，事务提交后（`TransactionSynchronizationManager.afterCommit`）批量调用 `minioClient.removeObject`。

---

## 四、外部服务客户端（Java 特有）

### 4.1 [MEDIUM] 预测/评估流 imageUrl 仍为相对路径，Python 服务无法访问

[SysPredLogServiceImpl.java:139-146](file:///e:/DehazeSystem/dehaze-java/src/main/java/com/pei/dehaze/service/impl/SysPredLogServiceImpl.java#L139) `resolveImageUrl` 返回 `/api/v1/files/download/{fileId}` 相对路径（代码中仍含 `// TODO: 注入 SysFileService 获取文件URL` 注释），Python 服务无法直接访问该 Java 内部路径。

**影响**：通过 fileId 触发的预测/评估流无法工作，只有外部传入 `imageUrl` 的链路能正常执行。

**改进建议**：注入 `SysFileService`，返回可被 Python 服务访问的完整 URL（或共享存储路径）。

---

## 五、数据访问（MyBatis 特有）

### 5.1 [MEDIUM] SysRoleMenuMapper.xml 中 `type` 列未限定表别名

[SysRoleMenuMapper.xml:35](file:///e:/DehazeSystem/dehaze-java/src/main/resources/mapper/SysRoleMenuMapper.xml#L35)：

```xml
AND type != '${@com.pei.dehaze.common.enums.MenuTypeEnum@BUTTON.getValue()}'
```

该 SQL 在 `sys_role_menu`/`sys_role`/`sys_menu` 三表 JOIN 查询中，`type` 列未限定表别名（应为 `t3.type`，因 `type` 列归属 `sys_menu` 表 t3），其余列均已加别名（`t2.code`、`t3.perm` 等），唯独 `type` 未限定，存在 `Column 'type' in where clause is ambiguous` 风险。

> 注：`${@...Enum@getValue()}` 字符串插值虽然绕过预编译缓存，但取的是静态枚举值，**无 SQL 注入风险**，本条仅针对表别名问题。

**改进建议**：限定表别名，如 `AND t3.type != ...`。

---

## 六、修复优先级清单

### P1（重要）

| # | 问题 | 文件 |
|---|------|------|
| 1 | 账户锁定机制完全缺失 | SysUserDetails.java, SysUser.java |
| 2 | 长事务仍包裹 MinIO（2 处） | DatasetOperationServiceImpl.java:312; SysItemFileServiceImpl.java:126 |
| 3 | 开发环境 JWT 密钥硬编码 fallback | application-dev.yml:110 |

### P2（改进）

| # | 问题 | 文件 |
|---|------|------|
| 4 | 预测/评估流 imageUrl 相对路径 | SysPredLogServiceImpl.java:139-146 |
| 5 | SysRoleMenuMapper.xml `type` 列未限定表别名 | SysRoleMenuMapper.xml:35 |
