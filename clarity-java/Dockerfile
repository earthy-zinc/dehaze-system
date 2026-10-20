# ==================== 三端单测回归用测试镜像 ====================
# 构建上下文 = 仓库根（--target test）：
#   docker build -f clarity-java/Dockerfile --target test -t csighub.tencentyun.com/earthywu/clarity-vision:java-test-<sha> .
# 为什么需要仓库根上下文：pom 的 copy-shared-sql 从 ../config/sql 取 schema/种子数据，
# 测试 resources 的 springdotenv.directory=../ 读仓库根 .env（运行时由 K8s Secret 挂载）。
# 运行态镜像（下方）不受影响，仍按原约定以 clarity-java/ 为上下文构建。
FROM maven:3.9-eclipse-temurin-21 AS test

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8

# Pod 内无 maven 私服凭据，统一走阿里云公共仓
# （不用 heredoc：Dockerfile 解析会吃掉不带 \ 续行的 heredoc 体，导致 settings.xml 为空）
RUN mkdir -p /root/.m2 \
    && printf '%s\n' \
      '<settings xmlns="http://maven.apache.org/SETTINGS/1.2.0">' \
      '  <mirrors>' \
      '    <mirror>' \
      '      <id>aliyun</id>' \
      '      <name>aliyun public</name>' \
      '      <url>https://maven.aliyun.com/repository/public</url>' \
      '      <mirrorOf>central</mirrorOf>' \
      '    </mirror>' \
      '  </mirrors>' \
      '</settings>' > /root/.m2/settings.xml \
    && cat /root/.m2/settings.xml

# 容器内布局与宿主检出路径无关：/app/clarity-java + /app/config + /app/.env（Secret 挂载）
WORKDIR /app/clarity-java

# 依赖层前置：pom 不变则依赖层命中缓存（源码变更只重建下面的源码层）
COPY clarity-java/pom.xml /app/clarity-java/pom.xml
RUN mvn -B -ntp -DskipTests dependency:resolve dependency:resolve-plugins

COPY clarity-java /app/clarity-java
COPY config /app/config

CMD ["mvn", "-B", "-ntp", "test"]

# ==================== 运行态镜像 ====================
FROM openjdk:17-jdk-alpine

# 设置国内镜像源(中国科技大学镜像源)，修改容器时区(alpine镜像需安装tzdata来设置时区)，安装字体库(验证码)
RUN echo -e https://mirrors.ustc.edu.cn/alpine/v3.7/main/ \
    > /etc/apk/repositories  \
    && apk --no-cache add tzdata \
    && cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && apk --no-cache add ttf-dejavu fontconfig

WORKDIR /app

# 在运行时自动挂载目录为匿名卷，提高可移植性
VOLUME /app/logs
VOLUME /app/upload
VOLUME /app/dataset

# 将构建的 Spring Boot 可执行 JAR 复制到容器中，重命名为 app.jar
COPY target/clarity-java.jar app.jar

# 指定容器启动时执行的命令
CMD ["java", "-Djava.security.egd=file:/dev/./urandom", "-jar", "app.jar", "--spring.profiles.active=prod"]

# 暴露容器的端口
EXPOSE 80
