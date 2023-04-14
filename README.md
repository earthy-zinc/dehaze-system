# reading-note

Learning Notes for Full Stack Development

这是一个全栈开发学习笔记

[filename](./docs/%E5%B7%A5%E4%BD%9C%E6%95%88%E7%8E%87/Zotero.md ':include')

## 虚拟机配置指南

### docker环境搭建：

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
docker run  \
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

### jenkins搭建

```bash
docker run \
-di \
--name=jenkins \
-p 8000:8080 \
-v /opt/docker_volume/jenkins:/var/jenkins_home \
jenkins/jenkins:lts
```

### nginx搭建

```bash
docker run --name nginx \
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
