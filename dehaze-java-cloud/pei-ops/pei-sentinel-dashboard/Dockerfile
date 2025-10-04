# 贝尔实验室 Spring 官方推荐镜像 JDK下载地址 https://bell-sw.com/pages/downloads/
FROM bellsoft/liberica-openjdk-debian:17.0.11-cds
#FROM bellsoft/liberica-openjdk-debian:21.0.5-cds
#FROM findepi/graalvm:java17-native

LABEL maintainer="Lion Li"

RUN mkdir -p /pei/sentinel-dashboard/logs \
    /pei/skywalking/agent

WORKDIR /pei/sentinel-dashboard

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 JAVA_OPTS=""

EXPOSE 8718

ADD ./target/pei-sentinel-dashboard.jar ./app.jar

SHELL ["/bin/bash", "-c"]

ENTRYPOINT java -Djava.security.egd=file:/dev/./urandom \
           #-Dskywalking.agent.service_name=pei-sentinel-dashboard \
           #-javaagent:/pei/skywalking/agent/skywalking-agent.jar \
           ${JAVA_OPTS} -jar app.jar
