# Nginx学习

## 一、Nginx配置

关于Nginx的配置，主要是以下五个方面：初始配置、基本语法、http服务配置、tcp/udp、反向代理

### 1、初始配置

```nginx configuration
# 启动的工作进程数
worker_processes 1;

# 设置每个工作进程的最大连接数，决定了nginx的并发能力
events {
    worker_connections 1024;
}


http {
    include mime.types;
    default_type application/octet-stream;
  
    sendfile on;
  
    # 决定一个请求完成之后还要保持连接多久
    # 目的是保持长连接，减少创建连接的过程给系统带来的性能损耗
    keepalive_timeout 65;

    server {
        # 监听的端口号
        listen 80;
        server_name localhost;

        # 匹配用户访问网站时的url /
        # 该配置指明当用户访问网站uri为"/"时，会访问到nginx的html目录下的index.html和index.htm文件
        location / {
            root html;
            index index.html index.htm;
        }

        # 指定出现错误的页面以及对应页面的存放位置
        error_page 500 502 503 504 /50x.html;
        location = /50x.html {
            root html;
        }
    }
}
```

工作进程：nginx在启动时会启动一个主要进程，该线程负责管理工作进程。而工作进程用来处理网络请求。worker_processes用于控制工作进程的数量。建议将工作进程数设置为系统cpu核心数，能够做到并行处理。

路径匹配：nginx对请求的客户端的返回是通过location来定义的，在之后的代码块中的return指令会直接返回响应状态码。之后的配置都是无效的。


### 2、基本语法

1. 配置文件由指令和指令块构成
2. 每个指令块以分号结尾
3. `include`语句允许组合多个配置文件
4. `#`为单行注释
5. `$`可以使用变量，`set`语句自定义变量


| 内置变量                 | 说明                                            |
|----------------------|-----------------------------------------------|
| $arg_name	           | 请求中的参数名                                       |
| $args	               | 请求中的参数值                                       |
| $binary_remote_addr	 | 客户端地址的二进制形式, 固定长度为4个字节                        |
| $body_bytes_sent	    | 传输给客户端的字节数，响应头不计算在内                           |
| $bytes_sent	         | 传输给客户端的字节数                                    |
| $content_length	     | “Content-Length” 请求头字段                        |
| $remote_addr         | 	客户端地址                                        |
| $remote_user	        | 用于 HTTP 基础认证服务的用户名                            |
| $request_body	       | 客户端的请求主体                                      |
| $request_length	     | 请求的长度 （包括请求的地址, http请求头和请求主体）                 |
| $request_method	     | HTTP 请求方法                                     |
| $request_time	       | 处理客户端请求使用的时间,从读取客户端的第一个字节开始计时                 |
| $request_uri	        | 这个变量等于包含一些客户端请求参数的原始 URI ，它不包含主机名             |
| $server_addr	        | 服务器端地址, 注意：为了避免访问 linux 系统内核，应将ip地址提前设置在配置文件中 |
| $status	             | HTTP 响应代码                                     |
| $time_local	         | 服务器时间                                         |
| $uri	                | 请求中的当前 URI, 不带请求参数，且不包含主机名                    |

### 3、http服务配置
#### 3. 1 listen指令
listen指令的上下文环境为server指令，作用为监听、拦截上层端口，处理该端口发来的请求

```nginx configuration
Syntax: 
    listen address[:port] [default_server] [ssl] [http2 | spdy] [proxy_protocol] [setfib=number] [fastopen=number] [backlog=number] [rcvbuf=size] [sndbuf=size] [accept_filter=filter] [deferred] [bind] [ipv6only=on|off] [reuseport] [so_keepalive=on|off|[keepidle]:[keepintvl]:[keepcnt]];
    listen port [default_server] [ssl] [http2 | spdy] [proxy_protocol] [setfib=number] [fastopen=number] [backlog=number] [rcvbuf=size] [sndbuf=size] [accept_filter=filter] [deferred] [bind] [ipv6only=on|off] [reuseport] [so_keepalive=on|off|[keepidle]:[keepintvl]:[keepcnt]];
    listen unix:path [default_server] [ssl] [http2 | spdy] [proxy_protocol] [backlog=number] [rcvbuf=size] [sndbuf=size] [accept_filter=filter] [deferred] [bind] [so_keepalive=on|off|[keepidle]:[keepintvl]:[keepcnt]];
Default:
    listen *:80 | *:8000;
Context:  server;
```

#### 3. 2 server指令
```nginx configuration
server {
    # 监听端口
    listen       8089;
    server_name  localhost;

    # 指定资源根路径
    root /data/yum_source;
    # 打开目录浏览功能
    autoindex on;
    # 指定网站初始页，找index.html或者index.htm页面
    index index.html index.htm;
}

```

### 4、tcp和udp


### 5、反向代理
反向代理指的是nginx将客户端的针对某个端口、某个url的请求转发给内部网络的指定服务器，并且将结果返回给客户端。
又叫做七层反向代理
这样做可以隐藏内部服务器的部署情况，通过统一入口控制流量。还可以设置负载均衡，将用户请求均分给多个服务器。反向代理指令为`proxy_pass`，其上下文环境
为`location`
反向代理可以将客户端发来的Http协议的数据包转换成其他协议（ fastcgi 协议、uwsgi协议、grpc、websocket协议）的数据包。
`stream` 指令，也可以用来实现四层协议( TCP 或 UDP 协议)的转发、代理或者负载均衡。叫做四层反向代理
#### 5. 1 四层反向代理
#### 5. 2 七层反向代理

