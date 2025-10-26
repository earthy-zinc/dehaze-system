# dehaze-java 测试框架文档

## 概述

本文档描述了 dehaze-java 项目的自动化测试框架设计与实现，旨在提供高可维护、分层清晰、覆盖核心业务场景的测试体系。

## 技术栈

- **测试框架**: JUnit 5
- **Spring 测试**: Spring Boot Test, Spring Security Test
- **数据库**: MySQL 数据库（测试环境）
- **Mock 框架**: MockMvc, @MockBean
- **断言库**: JUnit 5 Assertions, Hamcrest
- **ORM**: MyBatis-Plus 3.5.5

## 测试架构

### 测试分层

```
测试层次结构：
├── 单元测试 (Unit Tests)
│   └── Service 层测试
├── 集成测试 (Integration Tests)
│   ├── Mapper 层测试（含数据权限）
│   └── Repository 层测试
└── Web 层测试 (Controller Tests)
    ├── 接口功能测试
    ├── 安全认证测试（JWT）
    └── 权限控制测试（RBAC）
```

### 目录结构

```
src/test/
├── java/com/pei/dehaze/
│   ├── base/
│   │   └── BaseTest.java              # 测试基类
│   ├── service/
│   │   └── SysUserServiceImplTest.java  # Service 单元测试示例
│   ├── mapper/
│   │   └── SysUserMapperTest.java      # Mapper 集成测试示例
│   └── controller/
│       └── FileControllerTest.java      # Controller 安全测试示例
└── resources/
    ├── application-test.yml             # 测试配置
    └── db/
        ├── schema-test.sql              # 数据库表结构
        └── data-test.sql                # 测试数据
```

## 核心配置

### 1. 测试配置文件 (application-test.yml)

### 2. 测试基类 (BaseTest.java)

```java
@ExtendWith(SpringExtension.class)
@SpringBootTest
@ActiveProfiles("test")
@Transactional
public abstract class BaseTest {
    // 所有测试类继承此基类
}
```

**功能特性**：

- `@SpringBootTest`: 加载完整的 Spring 应用上下文
- `@ActiveProfiles("test")`: 激活 test 配置文件
- `@Transactional`: 自动回滚事务，保证测试隔离性

## 测试示例

### 1. Service 单元测试

**文件**: `SysUserServiceImplTest.java`

**测试场景**：

- ✅ 用户分页查询
- ✅ 用户新增（成功/失败）
- ✅ 用户更新
- ✅ 用户删除（单个/批量）
- ✅ 密码修改
- ✅ 用户名重复校验
- ✅ 获取认证信息

**示例代码**：

```java
@Test
@DisplayName("新增用户 - 用户名重复")
void testSaveUser_DuplicateUsername() {
    // Given: 使用已存在的用户名
    UserForm userForm = new UserForm();
    userForm.setUsername("root");
    // ...
    
    // When & Then: 应该抛出异常
    assertThrows(IllegalArgumentException.class,
            () -> sysUserService.saveUser(userForm),
            "用户名重复应该抛出异常");
}
```

### 2. Mapper 集成测试（数据权限）

**文件**: `SysUserMapperTest.java`

**测试重点**: 验证 `@DataPermission` 拦截器的 SQL 拼接逻辑

**测试场景**：

- ✅ 全部数据权限（ROOT 角色）- 不拼接过滤条件
- ✅ 本部门数据权限（ADMIN 角色）- 拼接部门过滤
- ✅ 仅本人数据权限（GUEST 角色）- 拼接用户过滤
- ✅ 分页查询基础功能
- ✅ 关键字搜索

**示例代码**：

```java
@Test
@DisplayName("数据权限测试 - 全部数据（ROOT 角色）")
void testDataPermission_AllData() {
    // Given: 模拟 ROOT 角色用户（dataScope=1）
    mockSecurityContext(1L, "root", Set.of("ROOT"), 1, 1L);
    
    // When: 执行查询（应该不拼接任何过滤条件）
    IPage<UserBO> result = sysUserMapper.listPagedUsers(page, query);
    
    // Then: 验证结果（应该能查到所有用户）
    assertTrue(result.getTotal() >= 3, 
        "ROOT 角色应该能查看所有用户");
}
```

### 3. Controller 安全测试

**文件**: `FileControllerTest.java`

**测试场景**：

- ✅ JWT 认证测试（有效/无效/缺失 token）
- ✅ 角色权限测试（ADMIN/USER/GUEST）
- ✅ 文件上传功能（成功/空文件/超大文件）
- ✅ 文件下载功能（成功/不存在）
- ✅ 文件删除权限控制
- ✅ 批量操作测试

**示例代码**：

```java
@Test
@WithMockUser(username = "admin", roles = { "ADMIN" })
@DisplayName("文件上传 - 成功")
void testFileUpload_Success() throws Exception {
    // Given: 准备上传文件
    MockMultipartFile file = new MockMultipartFile(
            "file", "test.jpg", 
            MediaType.IMAGE_JPEG_VALUE,
            "test image content".getBytes());
    
    // When & Then: 执行上传并验证
    mockMvc.perform(multipart("/api/v1/files/upload")
            .file(file)
            .with(csrf()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.code").value("00000"));
}
```

## 数据权限测试

### RBAC 权限模型

项目实现了基于角色的数据权限控制（RBAC），通过 MyBatis 拦截器动态拼接 SQL 过滤条件。

**数据权限范围**：

1. **全部数据** (dataScope=1): ROOT 角色，无过滤条件
2. **自定义数据** (dataScope=2): 自定义部门列表
3. **本部门及子部门** (dataScope=3): 部门树过滤
4. **本部门数据** (dataScope=4): 当前部门过滤
5. **仅本人数据** (dataScope=5): 用户 ID 过滤

### 测试数据

**初始用户**（在 data-test.sql 中）：

- `root` (ID=1): ROOT 角色，全部数据权限
- `admin` (ID=2): ADMIN 角色，本部门数据权限
- `test` (ID=3): GUEST 角色，仅本人数据权限

**密码**（BCrypt 加密）：

```
原始密码: 123456
加密后: $2a$10$xVWsNOhHrCxh5UbpCE7/HuJ.PAOKcYAqRxD2CO2nVnJS.IAXkr5aq
```

## 运行测试

### 运行所有测试

```bash
# Maven
mvn clean test

# Gradle
./gradlew test
```

### 运行特定测试类

```bash
mvn test -Dtest=SysUserServiceImplTest
```

### 运行特定测试方法

```bash
mvn test -Dtest=SysUserServiceImplTest#testSaveUser_Success
```

### IDE 中运行

在 IntelliJ IDEA 或 Eclipse 中：

1. 右键点击测试类或方法
2. 选择 "Run Test" 或 "Debug Test"

## 测试最佳实践

### 1. 命名规范

- **测试类**: `{ClassName}Test.java`
- **测试方法**: `test{MethodName}_{Scenario}`
- **示例**: `testSaveUser_Success`, `testSaveUser_DuplicateUsername`

### 2. Given-When-Then 模式

```java
@Test
void testExample() {
    // Given: 准备测试数据
    UserForm form = new UserForm();
    form.setUsername("test");
    
    // When: 执行被测试方法
    boolean result = userService.saveUser(form);
    
    // Then: 验证结果
    assertTrue(result, "保存应该成功");
}
```

### 3. 使用 @DisplayName

```java
@Test
@DisplayName("新增用户 - 用户名重复应该抛出异常")
void testSaveUser_DuplicateUsername() {
    // 测试代码
}
```

### 4. 事务回滚

所有测试类继承 `BaseTest`，自动使用 `@Transactional` 注解，测试后自动回滚，确保数据隔离。

### 5. Mock vs 真实依赖

- **单元测试**: 使用 `@MockBean` 模拟依赖
- **集成测试**: 使用真实的 Spring Bean 和数据库

## 测试覆盖率

### 生成覆盖率报告

```bash
mvn clean test jacoco:report
```

报告位置: `target/site/jacoco/index.html`

### 目标覆盖率

- **Service 层**: ≥ 80%
- **Mapper 层**: ≥ 70%
- **Controller 层**: ≥ 75%
- **整体覆盖率**: ≥ 70%

## 总结

本测试框架提供了：

✅ **快速运行**: H2 内存数据库，秒级启动  
✅ **测试隔离**: 事务自动回滚，无状态测试  
✅ **真实环境**: 完整 Spring 上下文，真实业务逻辑  
✅ **安全测试**: JWT + 角色权限 + 数据权限  
✅ **易于维护**: 清晰分层，标准命名，良好文档  
✅ **CI 友好**: 可并行执行，稳定可靠  

## 参考资料

- [Spring Boot Testing Documentation](https://docs.spring.io/spring-boot/docs/current/reference/html/features.html#features.testing)
- [JUnit 5 User Guide](https://junit.org/junit5/docs/current/user-guide/)
- [Spring Security Test](https://docs.spring.io/spring-security/reference/servlet/test/index.html)
- [H2 Database](https://www.h2database.com/)
- [MyBatis-Plus Documentation](https://baomidou.com/)

---

**作者**: earthyzinc  
**更新时间**: 2025-10-22  
**版本**: 1.0.0
