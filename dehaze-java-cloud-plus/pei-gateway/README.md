# pei-gateway 网关模块

> 本模块的总体说明、需求与背景、技术栈、架构设计、核心实现详解等文档已迁移至 dehaze-doc：
>
> 👉 [Java 微服务增强版架构文档](../../dehaze-doc/docs/05-子项目实现/Java微服务增强版架构文档.md)
>
> 本 README 仅保留与部署运维相关的启动方式、Docker 部署和日志配置说明。

## 启动方式

通过 `GatewayServerApplication` 启动类启动服务：

```java
@SpringBootApplication
public class GatewayServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(GatewayServerApplication.class, args);
    }
}
```

## Docker 部署

使用提供的 Dockerfile 进行容器化部署：

```dockerfile
FROM eclipse-temurin:21-jre

RUN mkdir -p /pei-gateway
WORKDIR /pei-gateway
COPY ./target/pei-gateway.jar app.jar

ENV TZ=Asia/Shanghai JAVA_OPTS="-Xms256m -Xmx256m"

EXPOSE 48080

CMD ["sh", "-c", "exec java ${JAVA_OPTS} -Djava.security.egd=file:/dev/./urandom -jar app.jar"]
```

## 日志配置

日志配置在 `logback-spring.xml` 中定义，支持控制台和文件两种输出方式：

```xml
<configuration>
    <!-- 控制台 Appender -->
    <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
        <encoder class="ch.qos.logback.core.encoder.LayoutWrappingEncoder">
            <layout class="org.apache.skywalking.apm.toolkit.log.logback.v1.x.TraceIdPatternLogbackLayout">
                <pattern>${PATTERN_DEFAULT}</pattern>
            </layout>
        </encoder>
    </appender>
    
    <!-- 文件 Appender -->
    <appender name="FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <!-- 文件滚动策略和编码配置 -->
    </appender>
</configuration>
```
