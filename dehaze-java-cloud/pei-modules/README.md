# pei-modules 微服务业务模块集

> 本目录下 `pei-workflow` 模块的概述、8 大功能详解、包结构、技术栈、架构图等文档已迁移至 dehaze-doc：
>
> 👉 [Java 微服务版架构文档](../../dehaze-doc/docs/05-子项目实现/Java微服务版架构文档.md)
>
> 本 README 仅保留模块定位、启动方式与接口文档说明。

## 模块定位

`pei-modules` 是 dehaze-java-cloud 的 **业务模块集合目录**，包含 `pei-workflow`（工作流引擎）等业务子模块。各子模块基于 Spring Boot + Dubbo + MyBatis Plus 实现，通过 Nacos 注册到微服务体系。

## pei-workflow 启动方式

通过工作流模块对应的 Spring Boot 启动类启动服务（具体类名见子模块源码），启动前需先启动 Nacos、MySQL/PostgreSQL、Redis 等依赖中间件。

```bash
# 编译打包
mvn clean package -pl pei-modules/pei-workflow -am

# 运行
java -jar pei-modules/pei-workflow/target/pei-workflow.jar
```

启动后默认监听端口见 `pei-workflow/src/main/resources/application.yaml` 中的 `server.port` 配置。

## 接口文档

服务启动后，可通过网关或直连方式访问 Swagger 文档：

- API 文档地址：`/doc.html` 或 `/swagger-ui/index.html`（具体路径见 `springdoc` 配置）

主要接口分组：

| 模块       | 控制器                       | 说明                       |
|----------|---------------------------|--------------------------|
| 流程定义     | `FlwDefinitionController` | 流程定义的查询、新增、发布、导出         |
| 流程实例     | `FlwInstanceController`   | 流程实例的启动、终止、挂起、恢复         |
| 任务管理     | `FlwTaskController`       | 任务的签收、完成、退回、指派           |
| 流程分类     | `FlwCategoryController`   | 流程分类的 CRUD               |
| 示例流程     | `TestLeaveController`     | 请假流程示例（apply / audit 等）  |

详细的请求/响应字段请通过 Swagger 文档查看。
