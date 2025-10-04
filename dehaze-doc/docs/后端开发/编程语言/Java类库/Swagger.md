
# Swagger

## 1、SpringBoot 集成 Swagger

1. 添加对应的依赖
2. 新建一个配置类，添加@EnableSwagger2 和@Configuration 注解，打开并自定义配置 Swagger
3. 通过 `http://项目IP:端口/swagger-ui.html`访问 API 接口文档

[附：SpringBoot 集成 Swagger 详细教程](http://www.imooc.com/wiki/swaggerlesson/springbootswagger.html)

## 2、常用注解

Swagger 是为了解决企业中接口（api）中定义统一标准规范的文档生成工具。可以通过在代码中添加 Swagger 的注解来生成统一的 API 接口文档。注解主要有以下几种：

| 注解名称           | 使用地方         | 说明                            |
| ------------------ | ---------------- | ------------------------------- |
| @Api               | 类               | 描述后端 API 接口类级别上的说明 |
| @ApiOperation      | 方法             | 描述后端 API 接口的信息         |
| @ApiParam          | 方法、参数、字段 | 对方法、参数添加元数据          |
| @ApiModel          | 类               | 对类进行说明                    |
| @ApiModelPropery   | 方法、字段       | 对类的属性说明                  |
| @ApiIgnore         | 类、方法、参数   | Swagger 将会忽略这些            |
| @ApiImplicitParam  | 方法             | 单独请求的参数                  |
| @ApiImplicitParams | 方法             |                                 |

| 注解参数    | 类型     | 默认值 | 涉及注解      | 说明                 |
| ----------- | -------- | ------ | ------------- | -------------------- |
| value       | String   |        |               | 描述接口用途         |
| tags        | String[] |        |               | 接口分组             |
| notes       | String   |        | @ApiOpreation | 对接口做出进一步描述 |
| httpMethod  | String   |        | @ApiOpreation | 接口请求方法         |
| nickname    | String   |        | @ApiOpreation | 接口别名             |
| protocols   | String   |        |               | 接口使用的网络协议   |
| hidden      | Boolean  |        |               | 是否隐藏该接口       |
| code        | int      |        | @ApiOpreation | 接口返回状态码       |
| description |          |        | @Api          |                      |
| produces    |          |        | @Api          |                      |
| consumes    |          |        | @Api          |                      |

## 3、Swagger 配置

创建 Swagger 的配置代码如下：

```java
@EnableSwagger2
@Configuration
public Class Swagger2Config{
    @Bean
    public Docket createApiDoc(){
        return new Docket(DocumentationType.SWAGGER_2)
            .apiInfo(apiInfo())
            .select()
            .apis(RequestHandlerSelector.basePackage("your_package_name"))
            .paths(PathSelectors.any())
            .build();
    }
    private ApiInfo apiInfo(){
        return new ApiInfoBuilder()
            .title()
            .description()
            .version()
            .build();
    }
}
```

| 方法名      | 描述              |
| ----------- | ----------------- |
| title       | 填写 API 文档标题 |
| description | 填写 API 文档描述 |
| version     | 填写 API 文档版本 |
| bulid       | 创建 ApiInfo 实例 |

