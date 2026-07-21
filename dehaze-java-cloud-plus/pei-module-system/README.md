# pei-module-system 系统管理模块

> 本模块的模块概述、目录结构、关键包详解、工作流程、实现原理、使用示例等文档已迁移至 dehaze-doc：
>
> 👉 [Java 微服务增强版架构文档](../../dehaze-doc/docs/05-子项目实现/Java微服务增强版架构文档.md)
>
> 本 README 仅保留模块定位、启动方式与接口文档说明。

## 模块定位

`pei-module-system` 是 dehaze-java-cloud-plus 的 **系统管理模块**，为微服务架构下的用户、部门、角色、权限、社交登录、短信、邮件、租户、站内信等基础功能提供统一的管理能力。基于 Spring Boot 3.4 + Java 17 + MyBatis Plus + OAuth2 实现，已支持多租户数据隔离。

## 启动方式

通过 `SystemServerApplication` 启动类启动服务：

```java
@SpringBootApplication
public class SystemServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(SystemServerApplication.class, args);
    }
}
```

启动后默认监听端口见 `application.yaml` 中的 `server.port` 配置，需先启动 Nacos、MySQL、Redis 等依赖中间件。

## 接口文档

服务启动后，可通过网关或直连方式访问 Swagger 文档：

- 管理后台接口：`/admin-api/system/...`（默认前缀）
- API 文档地址：`/doc.html` 或 `/swagger-ui/index.html`（具体路径见 `springdoc` 配置）

主要接口分组：

| 模块     | 路径前缀                          | 说明                |
|--------|-------------------------------|-------------------|
| 用户管理   | `/admin-api/system/user`      | 用户 CRUD、密码重置、状态切换 |
| 角色管理   | `/admin-api/system/role`      | 角色 CRUD、菜单权限分配    |
| 部门管理   | `/admin-api/system/dept`      | 部门树 CRUD          |
| 租户管理   | `/admin-api/system/tenant`    | 租户与套餐管理           |
| 短信服务   | `/admin-api/system/sms`       | 短信模板、渠道、发送记录      |
| 邮件服务   | `/admin-api/system/mail`      | 邮箱账户、模板、发送记录      |
| 社交登录   | `/admin-api/system/social`    | 第三方登录授权、绑定        |
| 站内信    | `/admin-api/system/notify`    | 通知模板、消息发送与查询      |

详细的请求/响应字段请通过 Swagger 文档查看。
