## 📢 项目简介

基于 JDK 17、Spring Boot 3、Spring Security 6、JWT、Redis、Mybatis-Plus、Knife4j 构建的前后端分离图像去雾系统后端。包括用户管理、角色管理、菜单管理、部门管理、字典管理等多个功能。后端自动生成接口文档，支持在线调试，提高开发效率。

## 项目亮点

1. **防⽌重复提交请求**：利⽤ Spring AOP 切⾯注解和 Redisson 分布式锁，通过加锁并设置过期时间，来对重要接⼝防⽌前端请求重复提交
2. **⽤户输⼊及权限校验**：基于 RBAC 模型，实现细粒度的权限控制，涵盖接口方法和按钮级别。利⽤JWT SpringSecurity 和 Redis，通过⽤户 ID 查询存储在 Redis 中当前⽤户权限，从⽽判断是否准许放⾏，后端针对传⼊参数通过注解验证，提供安全、无状态、分布式友好的身份验证和授权机制，提⾼系统健壮性
3. **项⽬管理**：利⽤接⼝、枚举、泛型定义后端常量，通过继承、实现等⾯向对象⽅法统⼀后端响应结构体，构建全局系统异常处理器，区分开发和⽣产配置，提⾼开发效率和可维护性

## 🌺 相关工程代码
| Gitee                                                        | Github                                                        |
|--------------------------------------------------------------|---------------------------------------------------------------|
| [dehaze-front](https://gitee.com/earthy-zinc/dehaze_front)   | [dehaze-front](https://github.com/earthy-zinc/dehaze_front)   |
| [dehaze-python](https://gitee.com/earthy-zinc/dehaze_python) | [dehaze-python](https://github.com/earthy-zinc/dehaze_python) |

## 🌈 接口文档

- `knife4j` 接口文档：[http://localhost:8989/doc.html](http://localhost:8989/doc.html)
- `swagger` 接口文档：[http://localhost:8989/swagger-ui/index.html](http://localhost:8989/swagger-ui/index.html)
- `apifox`  在线接口文档：[https://www.apifox.cn/apidoc](https://www.apifox.cn/apidoc/shared-195e783f-4d85-4235-a038-eec696de4ea5)


## 🚀 项目启动

1. **数据库初始化**

   执行 [dehaze.sql](sql/init.sql) 脚本完成数据库创建、表结构和基础数据的初始化。

2. **修改配置**

    [application-dev.yml](src/main/resources/application-dev.yml) 修改MySQL、Redis连接配置；

3. **启动项目**

    执行 [SystemApplication.java](src/main/java/com/pei/dehaze/SystemApplication.java) 的 main 方法完成后端项目启动；

    访问接口文档地址 [http://ip:port/doc.html](http://localhost:8989/doc.html) 验证项目启动是否成功。
