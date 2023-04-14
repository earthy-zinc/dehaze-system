

# docker 练习

## 一、介绍

docker使用客户端服务器架构，客户端通过命令行或其他工具与docker服务器进行通信。服务器用于执行用户命令和运行容器。

### 1、Docker仓库

镜像构建完成之后，如果要在其他服务器上使用这个镜像，我们就需要一个集中存储、分发镜像的服务。Docker Registry就是这样的服务。一个Docker Registry中可以包含多个仓库，每个仓库可以包含多个标签，每个标签对应一个镜像。通常一个仓库会包含同一个软件不同版本的镜像，我们可以通过`<RepositoryName>:<Tag>`的格式来指定具体是这个软件哪个版本的镜像，如果不给出标签，将以latest作为默认标签。

### 2、镜像

操作系统分为内核和用户空间，对于Linux而言，内核启动之后，会挂载root文件系统为其提供用户空间的支持。而docker镜像，就相当于一个root文件系统。docker镜像是一个特殊的文件系统，除了提供容器运行时所需的程序、库、资源、配置等文件外，还包含了一些为运行时准备的一些配置参数，镜像不包含任何动态数据，其内容在构建之后也不会被改变。

因为镜像是包含操作系统的完整root文件系统，其体积往往比较庞大，因此在docker设计时，就采用分层存储的架构，其实际体现是由一组文件系统组成。镜像在构建时，会一层层构建，前一层是后一层的基础。每一层构建完就不会再发生改变，后一层上的任何改变只发生在自己这一层。分层存储的特征还让镜像的复用、定制变得更为容易，甚至可以用之前构建好的镜像作为基础层，然后进一步添加新的层，以定制自己所需内容。

#### 1）虚悬镜像

在镜像中，有时我们能够看到一个特殊的镜像，没有仓库名、没有标签，都为`<none>`，这种镜像原来是由镜像名和标签的，但是随着官方镜像维护，重新发布了新的同版本镜像，镜像名就被转移到了新下载的镜像身上，而旧的镜像的名称责备取消。这类无标签的镜像也被称为虚悬镜像dangling image。

#### 2）中间层镜像

为了加速镜像构建、重复利用资源，Dockers会利用中间层镜像，所以镜像增多之后，就会看到一些无标签的中间层镜像，查看包含中间层在内的所有镜像，需要加-a参数。这些镜像如果被删除了会导致上层镜像出错。

### 3、容器

镜像和容器的关系，就和类和实例的关系一样。镜像是静态的定义，容器是镜像运行时的实体，容器可以被创建、启动、停止、删除、暂停

容器的实质时进程，但是与直接再宿主执行的进程不同，容器进程运行于属于自己的独立的命名空间。因此容器可以拥有自己的root文件系统、自己的网络配置、自己的进程空间，容器内的进程是运行再一个隔离的环境里，使用起来就好像是再一个独立于宿主的系统下的操作一样。

每一个容器运行时，是以镜像为基础层，在其上面创建一个当前容器的存储层，我们可以成这个为容器运行时读写而准备的容器存储层。容器不应该向其存储层写入任何数据，所有文件写入操作都应该使用数据卷，或者绑定宿主目录，在这些位置读写跳过容器存储层，数据卷的生存周期独立于容器，容器消亡，数据卷不会消亡，因此使用数据卷后，容器删除或者重新运行，数据不会丢失。

## 二、docker命令

### 1、基础命令

#### 1）systemctl

（system control）控制系统服务的运作

```shell
systemctl [command] [service]
```

* command 需要执行的命令，有start / stop / restart / enable / status，分别为开启服务、关闭服务、重启服务、自启动、查看服务状态
* service：要执行的服务，这里为docker

#### 2）docker

直接输入docker可以查询与docker有关的命令

```shell
docker [command] [options]
```

* command 需要执行的命令，有version / info / ，查询版本号信息
* options 可选项，--help 查看全部帮助，[command] --help 查看某条命令的帮助

```shell
# 查看镜像、容器、数据卷所占用的空间
docker system df
```



### 2）镜像命令

容器和镜像的关系类似于进程和程序的关系。当运行容器的时候，使用的镜像在本地不存在，docker会从镜像仓库中下载。

1. 列出本机镜像：`docker images ls`
   1. 列出虚悬镜像：`docker images ls -f dangling=true`
   2. 删除虚悬镜像：`docker image prune`
   3. 显示所有镜像：`docker image ls -a`
   4. 列出指定镜像：`docker image ls <image_name>[:version]`
   5. 列出按条件过滤后的镜像：`docker image ls -f[--filter] since=<image_name>[:version]`

2. 在仓库中查找镜像：`docker search <image_name>`
3. 拉取镜像到本地：
   1. `docker pull [options] [DockerRegistryAddress[:port]/]<image_name>[:version]`
   2. Docker 镜像仓库地址，格式一般是`<域名/IP>[:port]`，默认情况下地址为Docker Hub:`docker.io`
   3. image_name，也可以叫做RepositoryName，是两端式名称`<用户名>/<软件名>`对于Docker Hub:`docker.io`，如果不给出用户名，默认是library。

4. 删除本地镜像：`docker rmi <image_name>[:version]`
5. 更新本地镜像：
6. 生成镜像：
7. 为镜像添加标签：
8. 镜像导入
9. 镜像导出
10. 构建镜像：`docker build [options] <path/url/->`

### 3）容器命令

1. 运行容器：`docker run [options] image [command] [arg..]`

    | 命令简称 | 命令全称      | 参数类型 | 参数模板                               | 说明                                                         |
    | -------- | ------------- | -------- | -------------------------------------- | ------------------------------------------------------------ |
    | -d       | --detach      |          |                                        | 后台运行容器，并打印容器ID                                   |
    | -e       | --env         | list     |                                        | 设置环境变量                                                 |
    | -i       | --interactive |          |                                        | 允许交互式地运行容器（如果没有设为后台运行）                 |
    | -t       | --tty         |          |                                        | 打开一个伪终端(teletypewriter)，通常后面带有打开命令`/bin/bash` |
    | -p       | --publish     | list     | [outer_host]:[inner_host]              | 将容器的端口号映射到主机端口号                               |
    |          | --name        | string   | [container_name]                       | 为容器命名，名字是字符串类型                                 |
    | -v       | --volume      | list     | [host_directory]:[container_directory] | 为容器创建数据卷，同步容器内特定目录的数据到宿主机指定目录   |

    * **注：**我们可以通过`exit`命令或者使用`CTRL+D`退出容器
    * 当使用run命令创建容器时，docker在后台会进行检查
    * 检查本地是否存在指定镜像，不存在就从registry下载
    * 利用镜像创建并启动一个容器
    * 分配文件系统，在只读的镜像层外面挂载一层可读写层
    * 从宿主主机配置的网桥接口中桥接一个虚拟接口到容器中去
    * 从地址池配置一个ip给容器
    * 执行用户指定的应用程序

2. 查看容器：`docker ps [options]`

      *  options:
      *  -a --all 显示目前所有的容器（默认情况下只显示正在运行的）

3. 重启容器（适用于已停止但未删除的容器）：`docker start [container_id]`

4. 停止运行容器：`docker stop [container_id]`

5. 后台模式容器转前台运行：

      1. `docker attach [container_id]` 退出容器后会结束容器运行
      2. `docker exec [container_id]` 退出容器后不会导致容器停止

6. 容器的导出：`docker export [container_name] > [filename]`

7. 容器的导入：`docker import [filename] [image]:[tag]`

8. 强制停止容器：`docker rm [options] [container_name]`

9. 清理停止的容器：`docker container prune`

10. 查看容器运行日志：`docker logs [options]`

11. 在容器和本地宿主系统之间复制文件：`docker cp`

12. 重命名一个容器：`docker rename`

### 4）仓库管理

DockerHub是Docker官方维护的公共仓库。

1. 登录DockerHub：`docker login`
2. 退出登录：`docker logout`
3. 拉去镜像：`docker search imageName`
4. 下载镜像：`docker pull imageName`
5. 推送镜像到仓库：`docker push`

## 三、docker数据管理

### 1、数据卷volume的概念

数据卷是docker中用来持久化数据的概念，是一个可供一个或多个容器使用的特殊目录，里面存放有容器运行时产生的重要数据。它的生命周期独立于容器，即使容器被删除也不会删除容器对应的数据卷。并且也没有自动的垃圾回收机制会删除没有隶属于任何容器的数据卷。因此有时候会产生一些无主的数据卷，我们要通过命令来清理。数据卷有以下几个特点：

* 数据卷可以在容器之间共享和重用
* 对数据卷的修改会立刻生效
* 对数据卷的更新不会影响到容器
* 数据卷默认会一直存在，独立与docker容器，即使容器被删除

## 四、docker安装软件

### 1、MySql

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

mkdir -p /opt/docker_volume/mysql/log  && chown -R 200 /opt/docker_volume/mysql/log
mkdir -p /opt/docker_volume/mysql/data  && chown -R 200 /opt/docker_volume/mysql/data
mkdir -p /opt/docker_volume/mysql/conf  && chown -R 200 /opt/docker_volume/mysql/conf

docker run -d -p 3306:3306 --privileged=true -v /opt/docker_volume/mysql/log:/var/log/mysql -v /opt/docker_volume/mysql/data:/var/lib/mysql -v /opt/docker_volume/mysql/conf:/etc/mysql/conf.d -e MYSQL_ROOT_PASSWORD=123456 --name mysql mysql
```

### 2、redis

```bash
docker run \
-d \
-p 6379:6379 \
--name redis \
--privilege=true \
-v /opt/docker_volume/redis/redis.conf:/etc/redis/redis.conf \
-v /opt/docker_volume/redis/data:/data \
redis \
redis-server /etc/redis/redis.

```

### 3、jenkins

```bash
docker run -di --name=jenkins -p 8000:8080 -v /opt/docker_volume/jenkins:/var/jenkins_home jenkins/jenkins:lts
```

### 4、nacos

```bash
# 创建 nacos 配置存放目录
mkdir -p /opt/docker/nacos/conf  && chown -R 200 /opt/docker/nacos/conf

# 创建 nacos 日志存放目录
mkdir -p /opt/docker/nacos/logs  && chown -R 200 /opt/docker/nacos/logs

# 创建 nacos 数据存放目录
mkdir -p /opt/docker/nacos/data  && chown -R 200 /opt/docker/nacos/data

docker run -d -p 8848:8848 \
--name nacos \
--env MODE=standalone \
--env SPRING_DATASOURCE_PLATFORM=mysql \
--env MYSQL_SERVICE_HOST=rm-bp1035514z9uq43q46o.mysql.rds.aliyuncs.com \
--env MYSQL_SERVICE_PORT=3306 \
--env MYSQL_SERVICE_DB_NAME=mall_nacos \
--env MYSQL_SERVICE_USER=admin_1066365803 \
--env MYSQL_SERVICE_PASSWORD=142536aA \
nacos/nacos-server:latest

docker cp -a nacos:/home/nacos /opt/docker/

```

### redis

```shell
mkdir -p /opt/docker/redis

mkdir -p /opt/docker/redis/data

docker run --restart=always \
--log-opt max-size=100m \
--log-opt max-file=2 \
-p 6379:6379 \
--name redis \
-v /opt/docker/redis/redis.conf:/etc/redis/redis.conf \
-v /opt/docker/redis/data:/data \
-d redis redis-server /etc/redis/redis.conf  \
--appendonly yes  \
--requirepass 142536aA

docker cp redis:/etc/redis/ /opt/docker/redis/
```

### nginx

```shell
mkdir -p /opt/docker/nginx/html
mkdir -p /opt/docker/nginx/conf

docker run --name nginx \
-p 7788:80 \
-p 8911:81 \
-p 8912:82 \
-p 8913:83 \
-v /opt/docker/nginx/html:/usr/share/nginx/html \
-v /opt/docker/nginx/conf/nginx:/etc/nginx \
-d nginx

docker cp nginx:/etc/nginx /opt/docker/nginx/conf
```

### elasticsearch

```bash
mkdir -p /opt/docker/elasticsearch/config
mkdir -p /opt/docker/elasticsearch/data
echo "http.host: 0.0.0.0">>/opt/docker/elasticsearch/config/elasticsearch.yml

docker run --name elasticsearch -p 9200:9200 -p 9300:9300 \
-e "discovery.type=single-node" \
-e ES_JAVA_OPTS="-Xms256m -Xmx512m" \
-v /opt/docker/elasticsearch/config/elasticsearch.yml:/usr/share/elasticsearch/config/elasticsearch.yml \
-v /opt/docker/elasticsearch/data:/usr/share/elasticsearch/data \
-v /opt/docker/elasticsearch/plugins:/usr/share/elasticsearch/plugins \
-d elasticsearch:7.6.2

docker run --name kibana -e ELASTICSEARCH_HOSTS=http://119.96.97.103:9200 -p 5601:5601 -d kibana:7.6.2
```

### RabbitMQ

```bash
docker run -d --name rabbitmq \
-p 5671:5671 \
-p 5672:5672 \
-p 4369:4369 \
-p 25672:25672 \
-p 15671:15671 \
-p 15672:15672 \
rabbitmq:management
```

```bash
# 配置docker镜像加速
systemctl enable docker
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://o65lma2s.mirror.aliyuncs.com"]
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker

# 开机自启docker
systemctl enable docker

# 配置dockers kubernetes yum源为阿里云的
cat > /etc/yum.repos.d/kubernetes.repo << EOF
[kubernetes]
name=Kubernetes
baseurl=https://mirrors.aliyun.com/kubernetes/yum/repos/kubernetes-el7-x86_64
enabled=1
gpgcheck=0
repo_gpgcheck=0
gpgkey=https://mirrors.aliyun.com/kubernetes/yum/doc/yum-key.gpg
https://mirrors.aliyun.com/kubermetes/yum/doc/rpm-package-key.gpg
EOF

yum install -y kubelet-1.17.3 kubeadm-1.17.3 kubectl-1.17.3

systemctl enable kubelet
systemctl start kubelet


kubeadm init \
--apiserver-advertise-address=192.168.174.101 \
--image-repository registry.cn-hangzhou.aliyuncs.com/google_containers \
--kubernetes-version v1.17.3 \
--service-cidr=10.96.0.0/16 \
--pod-network-cidr=10.244.0.0/16

```

### 私有仓库

```bash
docker run -d -v /opt/docker/registry:/var/lib/registry -p 5000:5000 --name dockerRegistry registry
```

### jenkins

```bash
docker run -d -p 50001:8080 -p 50000:50000 \
-v /opt/docker/jenkins_home:/var/jenkins_home \
-v  /usr/share/maven:/usr/local/maven   -v /usr/libexec/git-core:/usr/local/git \
-v /etc/localtime:/etc/localtime \
--name jenkins \
jenkins/jenkins:lts
```



## 五、Dockerfile

镜像的定制实际上就是定制每一层所添加的配置、文件。如果我们可以把每一层修改、安装、构建、操作的命令都写入一个脚本，用这个脚本来构建、定制镜像。

Dockerfile是一个文本文件，内部包含了一条条指令，每一条指令构建一层，因此每条指令的内容，就是描述该层应当如何构建。

### 镜像构建上下文

dockers build命令最后有一个`.`而它表示当前目录，这里实际上是在指定上下文路径。

Docker在运行是分为Docker引擎和客户端工具，Docker引擎提供了一组Rest API被称为Docker Remote API，dockers命令是通过这组API和Docker引擎交互，从而完成各种功能呢，表面上我们在本机执行docker功能，实际上都是使用远程调用形式在服务端完成。

当我们进行镜像构建的时候，并非所有定制都会通过RUN命令完成，经常会需要把一些本地文件复制进镜像。这就引入了上下文的概念，当构建的时候，用户会指定构建镜像上下文路径，docker build命令得知这个路径后，会将路径下的所有内容打包，然后上传给docker引擎。docker引擎收到这个上下文包后，就会获得构建镜像所需的各种文件。

`COPY ./package.json /app/`并不是要复制执行docker build命令所在目录下的package.json，也不是复制Dockerfile所在目录下的，而是复制当时传入的上下文目录下的。因此COPY指令中的路径都是针对上下文路径上的相对路径。超出上下文路径或者是绝对路径的话，docker是无法获取这些文件的。

一般来说，应该将Dockerfile置于空目录下，或者项目根目录下，如果不希望目录下的有些东西在构建时传入Docker引擎，则需要写一个.dockerignore文件。

docker build还可以选择通过其他渠道构建镜像。如：

1. 使用git repo构建：`docker build -t hello-world https://github.com/docker-library/hello-world.git#master:amd64/hello-world`
2. 使用给定的tar压缩包构建：`docker build http://server/context.tar.gz`
3. 从标准输入中读取Dockerfile进行构建：`docker build - < Dockerfile` `cat Dockerfile | docker build -`
4. 从标准输入中读取上下文压缩包进行构建：`docker build - < context.tar.gz`

### FROM 指定基础镜像

FROM关键字指定基础镜像。因此Dockerfile中FROM时第一条指令，且是必须的。

### RUN 运行命令

RUN指令使用来执行命令行命令的，其格式通常由两种：

* shell格式：`RUN <command>`
* exec格式：`RUN ["可执行文件", "参数1", "参数2", ...]`

但是Dockerfile中每一条指令都会建立一层镜像。新建立一层，在这一层上执行命令，执行完命令后，通过commit命令提交这一层的修改，构成新的镜像。因此我们对于某些目的相同的指令，写在同一个RUN命令之内，不同的命令之间通过&& 连接执行。如果每一个命令都使用RUN命令，那么就会产生非常臃肿的镜像。

Dockerfile支持shell类命令的行尾添加`\`进行换行，以及行首使用`#`进行注释的格式。注意在目的达成之后，还需要添加清理工作的命令，清理下载、展开的文件和缓存，因为镜像是多层存储的，每一层的东西并不会在下一层被删除，因此在构建镜像时，要确保每一层只添加真正需要的东西，剩下的都要清理掉。

### COPY 复制文件

```bash
COPY [--chown=<user>:<group>] <源路径> <目标路径>
COPY [--chown=<user>:<group>] ["源路径", ...,"目标路径"]
```

和RUN指令一样，一种类似于命令行，一种类似于函数调用。

这个指令将从构建上下文目录中<源路径>的文件或目录中复制到新的一层镜像的目标路径位置。源路径可以是多个，也可以是通配符。目标路径可以是容器内绝对路径，也可以是相对于工作目录的相对路径，目标路径如果不存在会自动创建

使用COPY指令，源文件的各种源数据都会保留，如读写执行权限，文件变更时间。--chown选项用来改变文件所属用户以及所属组。

### ADD 复制文件

在COPY指令的基础上增加了一些功能。

源路径是URL，这种情况下Dockers引擎会试图去下载这个链接文件放到目标路径中去，并权限设置为600。如果更改权限、下载的是压缩包则需要额外的RUN指令进行调整。

如果源路径是压缩文件，ADD会自动解压这个压缩文件到目标路径中。推荐仅在这个场景下使用ADD命令

### CMD 容器启动

```shell
CMD <命令>
CMD ["可执行文件","参数1","参数2等"]
# 参数列表在制定了ENTRYPOINT质量后，用CMD指定具体参数
CMD ["参数1","参数2等",...]
```

在启动容器是，需要指定所运行的程序以及参数，CMD命令用于指定默认的容器主进程的启动命令，而且在运行是可以指定新的命令来替代镜像设置中的这个默认命令，在指令格式上，一般推荐使用exec格式，这类格式在解析时会被解析为JSON数据。如果使用shell格式，命令会被包装为sh -c的参数形式执行。

因此容器中应用需要前台运行。

### ENTRYPOINT 入口点

ENTRYPOINT的格式和RUN指令格式一样，目的和CMD一样，都是指定容器启动程序及参数，ENTRYPOINT在运行是也可以替代。当指定了入口点后，CMD的含义就发生了改变，不是直接运行其命令，而是将CMD的内容作为参数传给ENTRYPOINT，实际执行时，就变成了`<ENTRYPOINT> <CMD>`，那么为什么要使用ENTRYPOINT呢，因为在实际运行镜像时，我们可能也想传入一些参数，那么就会覆盖原有的CMD命令，有时我们既想保留CMD中的命令，并且运行镜像时增加参数。就要使用ENTRYPOINT来代替CMD，那么运行时传入的参数会替代CMD作为参数传给ENTRYPOINT。

第二是应用程序运行前要做的准备工作，如果我们需要在启动容器之前做一些事情，而不干扰到CMD指令，我们可以写一个脚本，放入ENTRYPOINT中执行。运行时可以往脚本中传入参数



### ENV环境变量

```shell
ENV <key> <value>
ENV <key1>=<value1> <key2>=<value2>
```

设置环境变量，后面的其他指令，或者时运行时的应用，都可以直接使用这里定义的环境变量。

### ARG构建参数

```shell
ARG <参数名>[=<默认值>]
```

和ENV环境变量的效果一样，都是设置环境变量，在ARG这里叫做参数。但是ARG设置的构建环境的环境变量在将来容器运行时是不会存在这些环境变量的。ARG指令中定义的参数值可以在构建命令中使用`--build-arg <>=<>`来覆盖。ARG中的参数又生效范围，如果在FROM之前指定，只能作用于FROM指令

### VOLUME 定义匿名卷

```shell
VOLUME ["路径1","路径2","",...]
VOLUME <路径>
```

容器在运行是应该尽量保持容器存储层不发生写操作，对于数据库类需要保存动态数据的应用，其文件应该保存到卷中，为了防止运行时用户忘记将动态文件所保存的目录挂载为卷，我们可以实现指定某些目录挂载为匿名卷。

### EXPOSE 声明端口

`EXPOSE <端口1>`

声明容器运行时提供服务的端口，只是帮助镜像使用者理解这个镜像服务的守护端口，以方便配置映射，在运行时使用随机端口映射时，会随机映射EXPOSE中声明的端口。

### WORKDIR 指定工作目录

`WORKDIR <workdir>`指定工作目录，在Shell中，连续两行时同一个进程的执行环境，因此前一个命令修改的内存状态，会直接影响后一个指令，而在Dockerfile中不同RUN命令的执行环境不同，如果需要改变以后各层工作目录的位置，让每一层的容器工作目录一致，就应该使用这一条指令。

如果当前WORKDIR指令使用的是相对路径，则所切换的路径于之前定义的WORKDIR有关

### USER 指定当前用户

### HEALTHCHECK 健康检查

### LABEL

### SHELL

指定RUN ENTRYPOINT CMD指令中使用的shell

## 六、数据管理

数据卷是一个可以供一个或者多个容器使用的特殊目录，它绕过UFS可以在容器之间共享和重用，对数据卷的修改会立刻生效，对数据卷的更新，不会影响到镜像，数据卷默认会一致存在即使容器被删除。镜像中被指定为挂载点的目录中的文件当数据卷为空的时候会复制进去

1. 创建数据卷 `docker volume create [volumeName]`
2. 查看所有的数据卷 `docker volume ls`
3. 查看指定数据卷的信息 `docker volume inspect [volumeName]`
4. 启动一个挂载数据卷的容器 使用--mount标记将数据卷挂载到容器里，可以一次挂载多个数据卷
5. 查看容器的具体信息，也可以查看到相关的数据卷绑定 `docker inspect [container]`
6. 删除数据卷 `docker volume rm [volumeName]`

## 七、网络

### 1、映射端口

容器中可以运行一些网络应用，当直接使用-P参数时，Docker会随机映射一个端口到内部容器开放的网络端口上。使用-p则可以指定要映射的端口，并且在一个指定端口上只能绑定一个容器。

使用ip:hostPort:containerPort可以将主机的对应端口映射到容器的对应端口

使用ip::containerPort绑定主机任意端口到容器的对应端口。还可以使用udp标记来指定udp端口。

使用docker port来查看当前映射的端口配置。容器有自己的内部网络和IP地址。

### 2、容器互联

1. 创建新的docker网络 `docker network create -d bridge net`
   1. docker的网络类型又bridge和overlay
2. 容器加入到网络  `docker run -it --rm --name contain1 --network net busybox sh`

### 3、配置DNS

自定义容器主机名和DNS需要通过Docker挂载容器的三个相关的配置文件，在容器中使用mount命令可以看到挂载的信息。这种机制可以在宿主机DNS信息更新后，docker容器DNS可以通过resolv.conf文件立刻更新。

配置所有容器的DNS可以在/etc/docker/daemon.json文件中增加DNS相关内容。

## 八、Docker Compose

对于Compose来说，大部分命令的对象既可以是项目本身，也可以指定为项目中的服务或者容器。

```shell
docker compose [-f=<arg>...] [options] [command] [args]
```

* -f, --file fileName 指定使用的Compose模板文件，默认为docker-compose.yml
* -p, --project-name Name 指定项目名称，默认将使用当前文件目录名称作为项目名
* --verbose 输出更多调试信息

1. build 构建命令，构建项目中的服务容器。
2. config 验证compose文件格式是否正确
3. down 停止up命令启动的容器，移除网络
4. exec 进入指定容器
5. images 列出compose文件中包含的镜像
6. kill 强制停止服务容器
7. logs 查看服务容器的输出
8. pause 暂停一个服务容器的运行
9. port 打印某个容器端口所映射的公共端口
10. ps 列出项目中目前所有的容器
11. pull 拉去服务容器依赖的镜像
12. push 推送服务依赖的镜像到仓库
13. restart 重启项目中的服务
14. rm 删除所有停止状态的容器
15. run 在指定服务容器中执行一个命令
16. start 启动一个已存在的容器
17. stop 停止一个运行中容器，但不删除
18. top
19. unpause
20. up 尝试自动完成构建镜像，创建服务，启动服务，关联服务相关容器的一系列操作

### Compose模板文件

每个服务都需要通过image指令指定镜像或者通过build指令来自动构建生成镜像
