# HTTP

## HTTP简介

HTTP（HyperText Transfer Protocol，超文本传输协议）是应用层协议，基于TCP/IP协议栈。HTTP采用请求-响应模型，由客户端发起会话请求，服务端返回响应。

## Socket协议

Socket是传输层协议的具体实现，封装了协议的复杂性，为开发人员提供了便利的网络连接接口。HTTP请求、数据库连接等都基于Socket实现。

### 传输层协议

Socket是传输层协议的软件实现，主要基于两种协议：

1. **TCP（Transmission Control Protocol）**
    - 面向连接的可靠传输协议
    - 具备顺序控制和重传机制
    - 建立连接需要三次握手，断开连接需要四次挥手

2. **UDP（User Datagram Protocol）**
    - 无连接的不可靠传输协议
    - 数据报文较小，传输速度快
    - 不需要预先建立连接

### Socket连接类型

1. **长连接**
    - 建立连接后保持通信状态
    - 双向自由传输数据
    - 数据传输完成后断开连接

2. **短连接**
    - 发起方建立连接
    - 发送数据后立即断开连接

长连接适用于数据传输量大、通信频繁的场景，但需注意服务器连接数限制。短连接适用于高并发请求场景，HTTP协议默认使用短连接。

### Socket编程示例

```java
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;

public class Server {
    public static void main(String[] args) {
        try (ServerSocket server = new ServerSocket(6000)) {
            // 等待客户端连接
            Socket socket = server.accept();
            // 向客户端发送信息
            OutputStream outputStream = socket.getOutputStream();
            outputStream.write("hello".getBytes());
            outputStream.flush();
            outputStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

public class Client {
    public static void main(String[] args) {
        try (Socket socket = new Socket("127.0.0.1", 6000)) {
            BufferedReader bufferedReader = new BufferedReader(
                    new InputStreamReader(socket.getInputStream())
            );
            System.out.println("客户端接收到: " + bufferedReader.readLine());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## CDN（内容分发网络）

CDN（Content Delivery Network）通过智能分配算法，将资源分发到离用户最近、带宽更高的服务器节点，改善用户体验。

### CDN工作原理

1. 用户请求域名，浏览器进行域名解析
2. 由于域名被CDN接管，解析获取到CNAME记录
3. CDN通过CNAME将访问代理到对应服务器
4. 浏览器获取最近CDN服务器IP地址
5. 浏览器访问CDN缓存服务器
6. CDN服务器判断资源是否存在或需要更新
7. 返回资源给用户

### CDN应用场景

1. **网页加速**：缓存HTML、CSS、JavaScript、图片等静态资源
2. **流媒体加速**：缓存视频、音频等大带宽资源
3. **文件下载加速**：提前将大文件分发到各地CDN节点
4. **边缘计算**：在靠近用户端进行数据处理，降低传输带宽
5. **网格化计算**：优化网络传输路径，提升访问速度

## HTTPS

### HTTP的不足

1. 明文传输，数据易被窃取
2. 缺乏通信双方身份验证
3. 无法保证数据完整性

### HTTPS介绍

HTTPS在HTTP基础上通过SSL/TLS建立安全信道，加密数据包。主要目的是：

- 提供网站服务器身份认证
- 保护交换数据的隐私和完整性

默认端口：

- HTTP：80
- HTTPS：443

### SSL/TLS协议

SSL（Secure Socket Layer）是安全套接字层协议，后被升级为TLS（Transport Layer Security）。

#### SSL/TLS提供的服务

1. 认证用户和服务器身份
2. 加密数据防止中途窃取
3. 维护数据完整性

#### SSL/TLS协议结构

SSL/TLS协议位于传输层和应用层之间，对应用层透明，包含：

1. **记录协议**：提供数据封装、压缩、加密等功能
2. **握手协议**：进行身份认证、协商加密算法、交换密钥等

### 加密算法

#### 对称加密算法

加密和解密使用相同密钥：

- 优点：速度快、计算量小
- 缺点：密钥传输不安全

#### 非对称加密算法

使用公钥和私钥配对：

- 公钥加密，私钥解密
- 私钥加密，公钥解密
- 优点：安全性高，无需传输密钥
- 缺点：算法复杂，效率低

#### 混合加密

结合两种算法优势：

1. 使用非对称加密协商对称加密密钥
2. 后续通信使用对称加密算法

### 数字证书

由权威认证机构颁发的身份认证证书，用于证明服务端或客户端身份。

#### 证书类型

1. **自签名证书**：自行生成的证书
2. **CA证书**：由认证机构颁发的证书

#### 服务端认证流程

1. 浏览器向服务器发送请求
2. 服务器发送证书和公钥
3. 浏览器验证证书签名
4. 验证通过后校验服务器身份
5. 进行密钥协商，建立安全通信

## 网络安全

### 常见攻击类型

#### 注入攻击

将恶意代码作为参数传入系统，诱使系统或用户触发。

常见类型：

- SQL注入
- 跨站脚本攻击（XSS）
- 命令注入

#### 欺骗攻击

伪造客户端身份访问服务器。

常见类型：

- 会话劫持
- 域名劫持
- IP欺骗

### 其他安全问题

1. **代码逻辑缺陷**：程序逻辑漏洞导致的安全问题
2. **服务器问题**：服务器配置不当或漏洞导致的安全风险
