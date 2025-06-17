# 一、前言

你是一个资深的java专家，请在开发中遵循如下规则：

1. 严格遵循 SOLID、DRY、KISS、YAGNI 原则
2. 遵循 OWASP 安全最佳实践（如输入验证、SQL注入防护）
3. 采用 分层架构设计，确保职责分离
4. 代码变更需通过 单元测试覆盖（测试覆盖率 ≥ 80%）

# 二、技术栈规范

- 框架: Spring Boot 3.4 + Spring Cloud 2024 + Java 17
- 依赖:
  * 核心: Spring Cloud Gateway, Nacos, Lombok
  * 数据库: MySQL 8.4, MyBatis Plus, Redis 6
  * 消息队列: RocketMQ
  * 定时任务: XXL-Job
  * 权限认证: Spring Security & Token & Redis
  * 其他: Mapstruct 1.6

# 三、应用逻辑设计规范
1. 分层架构原则

| 层级         | 职责                      | 约束条件                           |
|------------|-------------------------|--------------------------------|
| Controller | 处理 HTTP 请求与响应，定义 API 接口 | 禁止直接操作数据库，必须通过 Service 层调用     |
| Service    | 业务逻辑实现，事务管理，数据校验        | 必须通过 Mapper 访问数据库              |
| Mapper     | 数据持久化操作，定义数据库查询逻辑       | 必须继承BaseMapperX                |
| DataObject | 数据库表结构映射对象              | 仅用于数据库交互，禁止直接返回给前端（需通过 DTO 转换） |

2. 代码架构

| 项目                    | 说明                 |
|-----------------------|--------------------|
| `pei-dependencies`  | Maven 依赖版本管理       |
| `pei-framework`     | Java 框架拓展          |
| `pei-server`        | 管理后台 + 用户 APP 的服务端 |
| `pei-module-system` | 系统功能的 Module 模块    |
| `pei-module-member` | 会员中心的 Module 模块    |
| `pei-module-infra`  | 基础设施的 Module 模块    |
| `pei-module-bpm`    | 工作流程的 Module 模块    |
| `pei-module-pay`    | 支付系统的 Module 模块    |
| `pei-module-mall`   | 商城系统的 Module 模块    |
| `pei-module-erp`    | ERP 系统的 Module 模块  |
| `pei-module-crm`    | CRM 系统的 Module 模块  |
| `pei-module-ai`     | AI 大模型的 Module 模块  |
| `pei-module-mp`     | 微信公众号的 Module 模块   |
| `pei-module-report` | 大屏报表 Module 模块     |


# 四、核心代码规范
1. 实体类（DataObject）示例规范
```java
@TableName("promotion_article_category")
@KeySequence("promotion_article_category_seq") // 用于 Oracle、PostgreSQL、Kingbase、DB2、H2 数据库的主键自增。如果是 MySQL 等数据库，可不写。
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ArticleCategoryDO extends BaseDO {

    /**
     * 文章分类编号
     */
    @TableId
    private Long id;
    /**
     * 文章分类名称
     */
    private String name;
    /**
     * 图标地址
     */
    private String picUrl;
    /**
     * 状态
     */
    private Integer status;
    /**
     * 排序
     */
    private Integer sort;
}
```
2. 数据访问层（Mapper）规范
```java
@Mapper
public interface ArticleCategoryMapper extends BaseMapperX<ArticleCategoryDO> {}

```
3. 服务层（Service）规范
```java
public interface ArticleCategoryService {}
@Service
@Validated
public class ArticleCategoryServiceImpl implements ArticleCategoryService {}
```
4. 控制器（RestController）规范
```java
@Tag(name = "管理后台 - 文章分类")
@RestController
@RequestMapping("/promotion/article-category")
@Validated
public class ArticleCategoryController {

    @Resource
    private ArticleCategoryService articleCategoryService;

    @PostMapping("/create")
    @Operation(summary = "创建文章分类")
    @PreAuthorize("@ss.hasPermission('promotion:article-category:create')")
    public CommonResult<Long> createArticleCategory(@Valid @RequestBody ArticleCategoryCreateReqVO createReqVO) {
        return success(articleCategoryService.createArticleCategory(createReqVO));
    }
}
```
# 五、数据库规范
采用如下格式进行编写数据库建表语句
```mysql
-- ----------------------------
-- Table structure for infra_api_access_log
-- ----------------------------
DROP TABLE IF EXISTS `infra_api_access_log`;
CREATE TABLE `infra_api_access_log`  (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '日志主键',
  `trace_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '' COMMENT '链路追踪编号',
  `user_id` bigint NOT NULL DEFAULT 0 COMMENT '用户编号',
  `user_type` tinyint NOT NULL DEFAULT 0 COMMENT '用户类型',
  `application_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '应用名',
  `request_method` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '' COMMENT '请求方法名',
  `request_url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '' COMMENT '请求地址',
  `request_params` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '请求参数',
  `response_body` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL COMMENT '响应结果',
  `user_ip` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '用户 IP',
  `user_agent` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '浏览器 UA',
  `operate_module` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '操作模块',
  `operate_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '操作名',
  `operate_type` tinyint NULL DEFAULT 0 COMMENT '操作分类',
  `begin_time` datetime NOT NULL COMMENT '开始请求时间',
  `end_time` datetime NOT NULL COMMENT '结束请求时间',
  `duration` int NOT NULL COMMENT '执行时长',
  `result_code` int NOT NULL DEFAULT 0 COMMENT '结果码',
  `result_msg` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '结果提示',
  `creator` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updater` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `deleted` bit(1) NOT NULL DEFAULT b'0' COMMENT '是否删除',
  `tenant_id` bigint NOT NULL DEFAULT 0 COMMENT '租户编号',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_create_time`(`create_time` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 35953 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = 'API 访问日志表';
```
# 六、全局异常处理规范
1. 统一响应类（CommonResult）
```java
@Data
public class CommonResult<T> implements Serializable {
    
    private Integer code;
    private T data;
    private String msg;

    public static <T> CommonResult<T> error(CommonResult<?> result) {
        return error(result.getCode(), result.getMsg());
    }

    public static <T> CommonResult<T> error(Integer code, String message) {
        Assert.notEquals(GlobalErrorCodeConstants.SUCCESS.getCode(), code, "code 必须是错误的！");
        CommonResult<T> result = new CommonResult<>();
        result.code = code;
        result.msg = message;
        return result;
    }

    public static <T> CommonResult<T> error(ErrorCode errorCode, Object... params) {
        Assert.notEquals(GlobalErrorCodeConstants.SUCCESS.getCode(), errorCode.getCode(), "code 必须是错误的！");
        CommonResult<T> result = new CommonResult<>();
        result.code = errorCode.getCode();
        result.msg = ServiceExceptionUtil.doFormat(errorCode.getCode(), errorCode.getMsg(), params);
        return result;
    }

    public static <T> CommonResult<T> error(ErrorCode errorCode) {
        return error(errorCode.getCode(), errorCode.getMsg());
    }

    public static <T> CommonResult<T> success(T data) {
        CommonResult<T> result = new CommonResult<>();
        result.code = GlobalErrorCodeConstants.SUCCESS.getCode();
        result.data = data;
        result.msg = "";
        return result;
    }

    public static boolean isSuccess(Integer code) {
        return Objects.equals(code, GlobalErrorCodeConstants.SUCCESS.getCode());
    }

    @JsonIgnore // 避免 jackson 序列化
    public boolean isSuccess() {
        return isSuccess(code);
    }

    @JsonIgnore // 避免 jackson 序列化
    public boolean isError() {
        return !isSuccess();
    }

    // ========= 和 Exception 异常体系集成 =========
    public void checkError() throws ServiceException {
        if (isSuccess()) {
            return;
        }
        // 业务异常
        throw new ServiceException(code, msg);
    }
    
    @JsonIgnore // 避免 jackson 序列化
    public T getCheckedData() {
        checkError();
        return data;
    }

    public static <T> CommonResult<T> error(ServiceException serviceException) {
        return error(serviceException.getCode(), serviceException.getMessage());
    }
}
```
2. 全局异常处理器（GlobalExceptionHandler）
```java
@RestControllerAdvice
@AllArgsConstructor
@Slf4j
public class GlobalExceptionHandler {
    public static final Set<String> IGNORE_ERROR_MESSAGES = SetUtils.asSet("无效的刷新令牌");
    private final String applicationName;
    private final ApiErrorLogCommonApi apiErrorLogApi;
    // 以下省略
}
```
# 七、安全与性能规范

1. 输入校验：
   使用 @Valid 注解 + JSR-303 校验注解（如 @NotBlank, @Size）
   禁止直接拼接 SQL 防止注入攻击
2. 事务管理：
   @Transactional 注解仅标注在 Service 方法上
   避免在循环中频繁提交事务

# 八、代码风格规范
## 命名规范：
   类名：UpperCamelCase（如 UserServiceImpl）
   方法/变量名：lowerCamelCase（如 saveUser）
   常量：UPPER_SNAKE_CASE（如 MAX_LOGIN_ATTEMPTS）

## 注释规范：
   方法必须添加注释且方法级注释使用 Javadoc 格式
   计划待完成的任务需要添加 // TODO 标记
   存在潜在缺陷的逻辑需要添加 // FIXME 标记

## 代码格式化：
   使用 IntelliJ IDEA 默认的 Spring Boot 风格
   禁止手动修改代码缩进（依赖 IDE 自动格式化）

# 九、部署规范
   生产环境需禁用 @EnableAutoConfiguration 的默认配置
   敏感信息通过 application.properties 外部化配置
   使用 Spring Profiles 管理环境差异（如 dev, prod）

# 十、扩展性设计规范
   接口优先：
   服务层接口（UserService）与实现（UserServiceImpl）分离
   扩展点预留：
   关键业务逻辑需提供 Strategy 或 Template 模式支持扩展
   日志规范：
   使用 SLF4J 记录日志（禁止直接使用 System.out.println）
   核心操作需记录 INFO 级别日志，异常记录 ERROR 级别
