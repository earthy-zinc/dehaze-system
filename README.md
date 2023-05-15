# reading-note

Learning Notes for Full Stack Development

这是一个全栈开发学习笔记

## 虚拟机配置指南

### 硬件配置

| 节点名        | IP              | CPU  | 内存 |
| ------------- | --------------- | ---- | ---- |
| Pei-Linux-100 | 192.168.210.100 | 24   | 16   |
| Pei-Linux-101 | 192.168.210.101 | 4    | 8    |
| Pei-Linux-102 | 192.168.210.102 | 4    | 8    |
| Pei-Linux-103 | 192.168.210.103 | 4    | 8    |
|               |                 |      |      |



### docker环境搭建

#### 安装docker

使用官方给出的脚本，https://get.docker.com/

```shell
# 获取并运行安装脚本
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
# 启动docker服务
systemctl start docker
# 配置docker开机自启
systemctl enable docker
```

#### 配置国内镜像

修改配置文件

```shell
sudo vi /etc/docker/daemon.json
```

添加内容

```json
{
    "registry-mirrors": ["http://hub-mirror.c.163.com"]
}
```

重启docker

```shell
systemctl daemon-reload
systemctl restart docker
```

#### 注意事项：

如果以非 root 用户可以运行 docker 时，
需要执行 `sudo usermod -aG docker cmsblogs` 命令然后重新登陆，
否则会有如下报错

```md
docker: Cannot connect to the Docker daemon. 
Is the docker daemon running on this host ?
```

### mysql搭建

```bash
docker run --restart=always \
-d \
-p 3306:3306 \
--privileged=true \
-v /opt/docker_volume/mysql/log:/var/log/mysql \
-v /opt/docker_volume/mysql/data:/var/lib/mysql \
-v /opt/docker_volume/mysql/conf:/etc/mysql/conf.d \
-e MYSQL_ROOT_PASSWORD=123456 \
--name mysql \
mysql
```

### redis搭建

官网获取 `redis.conf` : http://www.redis.cn/download.html

解压后可在根目录查看到 `redis.conf` 配置文件，将其内容复制到 `/opt/docker_volume/redis/redis.conf`，然后修改以下内容：

- bind 127.0.0.1 ： 注释掉，redis可以外部访问
- --requirepass "123456" ： 设置Redis密码为123456，默认无密码
- --appendonly yes ： AOF持久化
- tcp-keepalive 60 ： 默认300，调小防止远程主机强迫关闭一个现有连接错误

```bash
docker run --restart=always \
--log-opt max-size=100m \
--log-opt max-file=2 \
-p 6379:6379 \
--name redis \
-v /opt/docker_volume/redis/redis.conf:/etc/redis/redis.conf \
-v /opt/docker_volume/redis/data:/data \
-d redis redis-server /etc/redis/redis.conf  \
--appendonly yes  \
--requirepass 123456
```

### rabbitmq搭建

```bash
docker run -d --restart always \
--name rabbitmq  \
--hostname rabbitmq \
-p 15672:15672 \
-p 5672:5672 \
rabbitmq
```

- hostname: RabbitMQ存储数据的节点名称,默认是主机名,不指定更改主机名启动失败,默认路径
- p 指定宿主机和容器端口映射（5672：服务应用端口，15672：管理控制台端口）

#### 安装插件

```bash
docker exec -it rabbitmq /bin/bash
rabbitmq-plugins enable rabbitmq_management
```

#### 验证

访问RabbitMQ控制台: [http://{host}:15672/](http://{host}:15672/)

用户名/密码：guest/guest

#### 重置队列

```bash
docker exec -it rabbitmq /bin/sh
rabbitmqctl stop_app
rabbitmqctl reset
rabbitmqctl start_app
```

### nacos搭建

#### 创建数据库



#### 创建和启动容器

```bash
docker run -d --name nacos --restart=always --network=host \
-e MODE=standalone \
-e JVM_XMS=256m \
-e JVM_XMX=512m \
-e SPRING_DATASOURCE_PLATFORM=mysql \
-e MYSQL_SERVICE_HOST=192.168.210.100 \
-e MYSQL_SERVICE_PORT=3306 \
-e MYSQL_SERVICE_DB_NAME=nacos \
-e MYSQL_SERVICE_USER=nacos \
-e MYSQL_SERVICE_PASSWORD=nacos \
-e MYSQL_DATABASE_NUM=1 \
-e MYSQL_SERVICE_DB_PARAM="characterEncoding=utf8&connectTimeout=1000&socketTimeout=3000&autoReconnect=true&useUnicode=true&useSSL=true&serverTimezone=UTC" \
-v /opt/docker_volume/nacos:/home/nacos/logs \
nacos/nacos-server:2.0.3
```

### seata搭建

#### 创建数据库

#### 配置

##### 1. 获取Seata外置配置

获取Seata外置配置在线地址：[config.txt](https://github.com/seata/seata/blob/1.5.2/script/config-center/config.txt)

##### 2. 导入外置配置

Nacos 默认**public** 命名空间下 ，新建Data ID 为 seataServer.properties 的配置，Group为SEATA_GROUP的配置，并将Seata外置配置config.txt内容全量复制进来

##### 3. 修改外置配置

seataServer.properties 需要修改存储模式为db和db连接配置

```properties
# 修改store.mode为db，配置数据库连接
store.mode=db
store.db.dbType=mysql
store.db.driverClassName=com.mysql.cj.jdbc.Driver
store.db.url=jdbc:mysql://192.168.210.100:3306/seata?useUnicode=true&rewriteBatchedStatements=true
store.db.user=root
store.db.password=123456
```

- **store.mode=db** 存储模式选择为数据库
- **store.db.url** MySQL主机地址
- **store.db.user** 数据库用户名
- **store.db.password 数据库密码

#### 创建和启动容器

##### 1. 获取应用配置

按照官方文档描述使用**自定义配置文件**的部署方式，需要先创建临时容器把配置copy到宿主机

**创建临时容器**

```bash
docker run -d --name seata-server -p 8091:8091 -p 7091:7091 seataio/seata-server:1.5.2
```

**创建挂载目录**

```bash
mkdir -p /opt/docker_volume/seata/config
```

**复制容器配置至宿主机**

```bash
docker cp seata-server:/seata-server/resources/ /opt/docker_volume//seata/config
```

注意复制到宿主机的目录，下文启动容器需要做宿主机和容器的目录挂载

**过河拆桥，删除临时容器**

```bash
docker rm -f seata-server
```

##### 2. 修改启动配置

在获取到 seata-server 的应用配置之后，因为这里采用 Nacos 作为 seata 的配置中心和注册中心，所以需要修改 application.yml 里的配置中心和注册中心地址，详细配置我们可以从 application.example.yml 拿到。

**application.yml 原配置**

**修改后的配置**(参考 application.example.yml 示例文件)，以下是需要调整的部分，其他配置默认即可

```yaml
seata:
  config:
    type: nacos
    nacos:
      server-addr: 192.168.10.99:8848
      namespace:
      group: SEATA_GROUP
      data-id: seataServer.properties
  registry:
    type: nacos
    preferred-networks: 30.240.*
    nacos:
      application: seata-server
      server-addr: 192.168.10.99:8848
      namespace:
      group: SEATA_GROUP
      cluster: default
```

- **server-addr** 是Nacos宿主机的IP地址，Docker部署别错填 localhost 或Docker容器的IP(172.17. * . *)
- **namespace** nacos命名空间id，不填默认是public命名空间
- **data-id: seataServer.properties** Seata外置文件所处Naocs的Data ID，参考上小节的 **导入配置至 Nacos**
- **group: SEATA_GROUP** 指定注册至nacos注册中心的分组名
- **cluster: default** 指定注册至nacos注册中心的集群名

##### 3. 启动容器

```bash
docker run -d --restart=always \
--name seata-server   \
-p 8091:8091 \
-p 7091:7091 \
-e SEATA_IP=192.168.10.100 \
-v /opt/docker_volume/seata/config:/seata-server/resources \
seataio/seata-server:1.5.2 
```

- **SEATA_IP：** Seata 宿主机IP地址

### minio搭建

```bash
docker run -d \
  --restart always \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio \
  -v /opt/docker_volume/minio/data:/data \
  -v /opt/docker_volume/minio/config:/root/.minio \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  quay.io/minio/minio server /data \
  --console-address ":9001"
```

- -e "MINIO_ROOT_USER=minioadmin" ： MinIO控制台用户名
- -e "MINIO_ROOT_PASSWORD=minioadmin" ：MinIO控制台密码

### jenkins搭建

```bash
docker run --restart=always \
-di \
--name=jenkins \
-p 8000:8080 \
-v /opt/docker_volume/jenkins:/var/jenkins_home \
jenkins/jenkins:lts
```

### nginx搭建

```bash
docker run --restart=always \
--name nginx \
-p 9000:9000 \
-p 9100:9100 \
-p 9101:9101 \
-p 9102:9102 \
-p 9103:9103 \
-p 9104:9104 \
-v /opt/docker_volume/nginx/html:/usr/share/nginx/html \
-v /opt/docker_volume/nginx/conf/nginx:/etc/nginx \
-d nginx
```
