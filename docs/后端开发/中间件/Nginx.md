# Nginx学习
本文主要讲解关于Nginx的配置，主要是以下五个方面：初始配置、基本语法、http服务配置、tcp/udp、反向代理

## 一、初始配置

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

## 二、基本语法

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
| request_filename     | 访问静态资源的完整路径                                   |
| document_root        | 静态资源文件所在的目录                                   |
| realpath_root        | 将软链接替换为真正的地址                                  |

## 三、http服务配置

### 3. 1 http请求处理

Nginx将一个Http请求分成了多个阶段，以模块为单位进行处理。总共11个阶段

#### POST_READ

Nginx在接收到Http请求头之后的处理。这里使用realip模块获取用户的真实地址，方便后续对该IP进行限速或者过滤请求

#### SERVER_REWRITE、REWRITE

这两个阶段使用rewrite模块修改Http请求的uri地址，实现对请求的控制

#### FIND_CONFIG

对location匹配

#### PRE_ACCESS 、ACCESS、 POST_ACCESS

对Http请求进行权限控制。 preaccess是在连接之前的访问控制，这个阶段limit_conn（并发连接限制）
和limit_req（每秒请求限制）等模块工作，
access阶段根据用户名密码等限制用户访问(auth_basic)、根据ip限制用户访问(access模块)以及第三方模块认证限制用户访问
post_access在access阶段之后需要做的工作

#### TRY_FILES

为访问静态资源设置的

#### CONTENT

处理http请求内容的阶段。index、autoindex、concat、反向代理模块在这里生效

#### LOG

处理完成请求之后的日志记录阶段

### 3. 2 realip模块

real ip即真实ip模块，该模块的作用是当本机的nginx处于反向代理后端时，获取到用户的真实ip。
这里的意思是根据http协议，在请求经过一次转发之后，http消息中就会增加转发服务器的相关信息如ip，
因此如果在之前我们对这条http消息已经做了反向代理，那么到达这个nginx服务器时，http消息中已经加入了
上一个转发服务器的ip。我们要再往前追溯才能找到用户真实的ip。

```nginx configuration
# 指定后端的代理服务器
set_real_ip_from 10.10.10.10;
# 设为off时，nginx会把header指定的Http头中的最后一个ip当成真实ip
# 设为on时，将header指定的http头中的最后一个不是信任服务器的ip当成真实的ip
real_ip_recursive off;
real_ip_header X-Forwarded-For;
```

### 3. 3 rewrite模块

该模块主要是为了改写请求uri，有时候用户请求的地址和我们后端服务器定义的地址不相同，那么nginx就可以根据一定的规则，
将uri改写为后端定义的地址，然后再转发给后端服务器处理。
rewrite模块会根据正则匹配重写uri，然后发起内部跳转在进行匹配location，或者直接做重定向返回客户端，
该模块的指令有break、if、return、rewrite、set

#### return指令

```nginx configuration
# code 指代http状态码，text为http消息体
Syntax: return code [text];
Default: —;
Context: server, location, if;
```

#### rewrite

```nginx configuration
Syntax:  rewrite regex replacement [flag];
Default: --
Context: server, location, if;
```

1. 将正则表达式指定的url替换成对应新的url，当替换的url以`http://` `https://`开头时，直接返回重定向
2. 替换后的url根据flag指定的方式进行处理

[flag]:

* last: 用replacement的url进行新的地址匹配
* break: 停止当前脚本指令的执行
* redirect:返回302重定向
* permanent: 返回301重定向

#### if

```nginx configuration
Syntax:  if (condition) { 
    # 脚本语句
 }
Default: —;
Context: server, location;
```

* 检查变量是否为空或者为 0
* 将变量与字符串做匹配，使用 = 或者 !=
* 将变量与正则表达式做匹配:
    * ~ 或者 !~ 大小写敏感
    * ~* 或者 !~* 大小写不敏感
* 检查文件是否存在 -f 或者 !-f
* 检查目录是否存在 -d 或者 !-d
* 检查文件、目录、软链接是否存在 -e !-e
* 是否为可执行文件 -x 或者 !-x

### 3. 4 location匹配

| 规则  | 说明                       |
|-----|--------------------------|
| =   | 严格匹配                     |
| ~   | 区分大小写匹配（可用正则）            |
| ~*  | 不区分大小写（可用正则）             |
| !~  | 区分大小写的不匹配                |
| !~* | 不区分大小写的不匹配               |
| ^~  | 前缀匹配                     |
| @   | 定义一个命名的location，使用在内部定向时 |
| /   | 通用匹配                     |

* `!`意味着不会匹配到某些url，在我们可以排除某些url的时候会有用
* `=`意味精准匹配，在匹配成功后，就会停止匹配，进行后续处理

### 3. 5 limit_conn模块

该模块限制单个ip建立的网络连接的个数，有六个指令：

* limit_conn_zone 分配共享内存
* limit_conn_status 拒绝的请求返回的http状态码
* limit_conn 指定限制单个ip请求的并发连接数
* limit_conn_log_level 达到最大限制连接数后，记录的日志的等级
* limit_conn_dry_run 设置演习模式，连接数不再受限制

### 3. 6 limit_req模块

用户处理突发流量，基于漏斗算法将突发的流量限定为恒定的流量，如果请求的容量没有超出设定的极限，后续的突发请求响应
会变慢，对于超过容量的请求会返回503

### 3. 7 allow deny指令

允许或者拒绝某些ip访问

```nginx configuration
Syntax: allow address | CIDR | unix: | all;
Default: —
Context: http, server, location, limit_except

Syntax: deny address | CIDR | unix: | all;
Default: —
Context: http, server, location, limit_except;
```

### 3. 8 auth_basic模块

基于Http基本认证协议进行的用户名和密码的认证

```nginx configuration
Syntax: auth_basic string | off;
# 默认是关闭的
Default: auth_basic off;
Context: http, server, location, limit_except

Syntax: auth_basic_user_file file;
Default: —
Context: http, server, location, limit_except;
```

### 3. 9 try_files指令

该指令一次访问多个uri对应的文件，当文件存在直接返回其内容，如果不存在，则按最后一个url结果或者http代码返回

```nginx configuration
Syntax: try_files file ... uri;
try_files file ... =code;
Default: —
Context: server, location;
```

### 3. 10 alias指令

```nginx configuration
# alias会将原来的地址替换，root不会
# alias 后面path必须要用“/”结束，否则会找不到文件的，而 root 则可有可无
Syntax: alias path
Default: —
Context: location;
```

### 3. 11 root指令

```nginx configuration
Syntax: root path
Default: root html
Context: http, server, location, if in location;
```

### 3. 12 listen指令

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

### 3. 13 server指令
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

### 3.14 log

```nginx configuration
log_format compression '$remote_addr - $remote_user [$time_local] '
                       '"$request" $status $bytes_sent '
                       '"$http_referer" "$http_user_agent" "$gzip_ratio"';

access_log /spool/logs/nginx-access.log compression buffer=32k;

# access_log指令用法
Syntax: access_log path [format [buffer=size] [gzip[=level]] [flush=time] [if=condition]];
access_log off;
Default: access_log logs/access.log combined;
Context: http, server, location, if in location, limit_except

# log_format指令用法
Syntax: log_format name [escape=default|json|none] string ...;
Default: log_format combined "...";
Context: http

# 是否打开日志缓存
Syntax: open_log_file_cache max=N [inactive=time] [min_uses=N] [valid=time];
open_log_file_cache off;
Default: open_log_file_cache off;
Context: http, server, location;
```

## 四、tcp和udp

## 五、反向代理

反向代理指的是nginx将客户端的针对某个端口、某个url的请求转发给内部网络的指定服务器，并且将结果返回给客户端。
又叫做七层反向代理
这样做可以隐藏内部服务器的部署情况，通过统一入口控制流量。还可以设置负载均衡，将用户请求均分给多个服务器。反向代理指令为`proxy_pass`
，其上下文环境
为`location`
反向代理可以将客户端发来的Http协议的数据包转换成其他协议（ fastcgi 协议、uwsgi协议、grpc、websocket协议）的数据包。
`stream` 指令，也可以用来实现四层协议( TCP 或 UDP 协议)的转发、代理或者负载均衡。叫做四层反向代理

### 5. 1 四层反向代理

转发TCP或UDP报文

### 5. 2 七层反向代理

将服务器对一个ip地址的请求转发到我们proxy_pass定义的那个IP地址中

```nginx configuration
# 在转发 http 请求时，URL必须以 http 或者 https 开头 
Syntax: proxy_pass URL;
Default: —
Context: location, if in location, limit_except;
```

如果URL不携带URI，会将对应的URL直接转发到上游服务器
如果携带URI，会将location参数中匹配上的那一段替换为该URL

## 六、负载均衡

用来在多个计算机、网络连接、CPU、磁盘中分配负载，以达到最优化资源使用、最大化
吞吐率、最小化响应时间、同时避免过载。也就是将负载进行平衡，分担到多个操作单元
以解决高性能、单点故障

### 6.1 负载均衡分类

负载均衡的分类可以按照网络层次进行分类。最常见的是四层和七层负载均衡

#### 二层负载均衡

负载均衡的服务器对外提供的是一个虚拟的IP，在内部的集群服务器中，不同的机器采用相同的IP地址，但是机器的MAC地址
不同，负载均衡服务器接收到请求后，通过改写报文的目标MAC地址的方式来将请求转发到目标机器上

#### 三层负载均衡

与二层类似，但是内部的集群服务器不同的机器采用的是不同的IP地址，负载均衡服务器在接收到请求之后，根据不同的负载均衡算法
通过IP将请求转发到真实的服务器。

#### 四层负载均衡

工作在OSI模型的传输层，在传输层只有TCP和UDP协议，这两种协议除了包含源IP和目的IP，还包含端口号，四层负载均衡服务器
在收到客户端的请求之后，通过修改数据包IP+端口号将请求转发到对应的服务器

#### 七层负载均衡

工作在OSI的应用层，应用层协议较多，由Http、DNS，同一个Web服务器的负载均衡，可以根据IP+端口，也可以根据URL、浏览器的类别、语言
、自定义参数来决定如何进行负载均衡。

### 6.2 负载均衡的算法

一类是静态的负载均衡算法，常见的由轮询、权重；另一类是动态的负载均衡算法，常见的有最少连接、最快响应。

### 6.3 负载均衡的配置

```nginx configuration
Syntax: upstream name { 
    server server_name;
}
Default: —
Context: http;
```

我们可以在server指令中定义服务器，可以用域名、ip+端口、socket形式指定地址，后面可以跟若干配置参数，默认情况下，upstream
指令块中采用的是加权Round-Robin负载均衡算法，该算法通过加权轮询的方式访问upstream中的server指令指定的上游服务。我们可以
指定服务的权重，最大并发连接数，最大失败数、超时时间等。
nginx还提供了其他的负载均衡算法，如基于客户端ip地址的Hash算法。该算法以客户端的IP地址最为hash算法的关键字，映射到特定的
上游服务器中，当然也可以根据客户端的其他key来进行哈希算法。
最少连接数算法，该算法从所有的上游服务器中找到并发连接数最少的一个，然后将请求转发给他，如果出现多个最少连接数
的服务器，则会在这些最少连接数的服务器中继续应用Round-Robin算法。配置该策略的指令是least_conn

## 七、缓存和压缩配置

## 八、日志

## 九、nginx基础架构

### 9.1 Nginx进程模型

Nginx在启动是会首先启动一个Master进程，然后由Master进程启动一个或多个Worker子进程，Master进程的工作
是完成配置读取、通过发送信号控制Worker进程的启动和停止，而Worker子进程是用来处理客户端发来的Http请求，Worker
进程之间会通过共享内存进行通信。

#### 工作进程处理请求的过程

假设Nginx启动了多个工作进程，并且在主进程中通过socket套接字监听80端口，这些工作进程是主线程的分支，每个工作
进程都可以监听端口，当有一个连接进来后，所有监听端口的工作进程都可以收到消息，但是只有一个进程可以接收这个连接，其他的
进程都会失败，nginx通过加锁来处理整个流程，在同一个时刻，只有一个进程可以接受连接。

#### 命令行处理流程

1. 首先主进程检查nginx.conf文件是否存在语法错误，并找到nginx.pid配置路径
2. reload参数表示想主进程发送HUP信号，Nginx根据nginx.pid找到主进程的pid

### 9.2 Nginx事件驱动模型

事件驱动模型

### 9.3 Nginx模块化设计

Nginx内部结构由核心部分和一系列功能模块所组成。这样的架构使得每个模块的功能相对简单，便于开发同时也便于对系统进行功能扩展
Nginx将各个功能模块组织成一条链，当有请求到达时，请求一次经过这条链上的部分或者全部模块。同时Nginx开放了第三方模块
的编写功能，用户可以自定义模块，控制请求的处理和响应。

#### Nginx的模块分类
