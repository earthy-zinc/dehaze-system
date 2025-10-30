# Web开发

## Servlet介绍

实际上要编写一个完善的http服务器需要耗费大量的时间，考虑许多东西。所以Java就提供了运行在Web服务器的程序叫做Servlet，一些底层的任务比如说识别错误正确的请求，解析http协议等交给它去做。我们使用Servlet提供的API来处理HTTP请求。

Servlet主要执行以下任务：

* 读取客户端发来的显式数据，主要是网页上的HTML表单产生的数据。
* 读取客户端发来的隐式请求数据，包括Cookies、媒体类型、浏览器能够理解的压缩格式等。
* 处理数据并生成结果。解析客户端发来的数据。
* 发送显式的数据到客户端，这些数据格式可以是多种多样的，包括文本文件，二进制文件等。
* 发送隐式的HTTP响应到客户端。比如说HTTP请求头的内容，设置Cookies、缓存参数、设置返回文档类型。

## Servlet生命周期

Servlet生命周期是Servlet对象从创建到毁灭的整个过程。

* Servlet 初始化后调用init()方法
* Servlet 调用 service()方法来处理客户端请求
* Servlet 销毁前调用 destroy()方法
* Servlet 由JVM的垃圾回收器回收它所占用的内存

### init()

一个Servlet对象创建于用户第一次调用该Servlet对应的URL时，只调用一次，后续每次用户请求时都不再调用。用户每次调用这个Servlet，都会创建一个实例对象，也就是说每一个用户请求都会产生一个新线程。init()
方法会简单的创建或者加载一些数据，这些数据将用于Servlet整个生命周期。

### service()

service()方法是执行实际任务的主要方法，容纳Servlet对象的那个容器（通产是Web服务器）会调用service()
方法处理客户端发来的请求，然后将格式化的响应发给客户端。service()方法会检查HTTP请求类型，判断是GET POST PUT
DELETE。然后调用doGet(), doPost() doPut() doDelete() 方法。

我们自己编写的Servlet需要继承`HttpServlet`，用户最常用的请求类型是Get Post，因此我们覆写`doGet() doPost()`方法。这两个方法传入了
`HttpServletRequest`和`HttpServletResponse`两个对象，分别代表HTTP请求和响应。这两个对象已经封装好了请求和响应。我们需要简单的获取请求参数，设置正确的响应类型，然后在写入响应即可。

#### `HttpServletRequest`

浏览器发来的HTTP请求都封装到了`HttpServletRequest`这个对象中，我们通过`HttpServletRequest`提供的方法可以拿到所有的HTTP请求信息。
`HttpServletRequest`从`ServletRequest`继承而来。通过`getXXX()`方法获取。

#### `HttpServletResponse`

服务器发送的HTTP响应需要封装到`HttpServletResponse`这个对象中，那么就需要设置响应头，通过`setXXX()`方法设置。

### destroy()

destroy()方法只会调用一次，可以在这里关闭数据库连接、停止后台线程、执行一些清理活动。

### 总结

实际上我们在web应用程序中并没有创建Servlet对象，也没有自己确定Servlet对象在何时会调用，没有实现服务器端和客户端之间的通信如TCP连接，解析HTTP协议的具体细节。因此对于web应用程序，我们必须先需要一个服务器来替我们做这些工作，再由服务器加载我们编写的Servlet，这样就可以让Servlet处理浏览器发送的请求。我们就需要找一个支持Servlet
API的Web服务器。

Tomcat是一个WEB服务器，也是由Java编写的，启动Tomcat服务器实际上启动了Java虚拟机，执行了Tomcat的main()
方法，然后Tomcat负责加载我们自己写的程序，创建一个Servlet实例，以多线程的模式来处理HTTP请求。

那么Tomcat服务器就是一个Servlet的容器。在容器中的Servlet有以下的特点：

* 无法在代码中直接通过new创建Servlet实例，必须由Servlet容器自动创建Servlet实例
* Servlet容器只会给每个Servlet类创建唯一实例
* Servlet会多线程的执行doGet() doPost()方法

## Servlet开发

### Dispatcher

一个web应用程序由一个或者多个Servlet组成的，每个Servlet通过注解说明自己能处理的URL路径。对于用户不同的请求路径要交给不同的Servlet处理。那么客户端发来的HTTP请求总是由WEB服务器先接收，然后根据Servlet配置的映射路径。不同的路径转发给不同的Servlet处理。那么就需要一个中间商来处理请求交个哪一个Servlet处理。这个中间商所实现的功能称为分发（Dispatch），中间商我们称为分发器（Dispatcher）。

分发器收到请求，判断路径，交给不同的Servlet，代码实现可以如下：

```java
//收到一个浏览器发来的路径 String path;
if(path.equals("hello")){
	dispatchTo(helloServlet);
}else if(path.equals("login")){
	dispathTo(loginServlet);
}else{
    dispatchTo(indexServlet);
}
```

### Redirect

重定向指的是当浏览器请求一个URI时，服务器返回一个 重定向指令，告诉浏览器地址已经变了，需要使用新的URI再次重新发送请求。

比如说我们已经编写了一个能处理路径为`/hello`的Servlet，如果收到的路径是`/hi`，我们希望让浏览器看到路径为`/hello`
的Servlet，那么再编写一个Servlet命名为`RedirectServlet`，在这个Servlet内部实现重定向到`/hello`。

如果浏览器发送`GET /hi`请求，`RedirectServlet`将处理此请求。由于`RedirectServlet`在内部又发送了重定向响应。浏览器会根据服务器发回的指示发送一个新的
`GET /hello`请求。整个过程浏览器发送了两次HTTP请求。

重定向有两种：一种是302响应，称为临时重定向。一种是301响应，称为永久重定向。对于永久重定向，浏览器会缓存`/hi`到`/hello`
这个重定向的关联，下次请求`/hi`的时候，浏览器就直接发送`/hello`请求了。

重定向的目的是当WEB应用升级后，如果请求路径发生了变化，可以将原来的路径重定向到新的路径，避免浏览器找不到在原路径上的资源。

### Forward

Forward是指内部转发。当一个Servlet处理请求的时候，它可以决定自己不继续处理，而是转发给另一个Servlet处理。对于浏览器来说，它只发出了一个HTTP请求。浏览器并不知道服务器在其内部做了一次转发。

### Session

对于要注册登录的应用，需要跟踪用户身份，服务器可以向浏览器分配一个唯一的ID，并用Cookies的形式发送到浏览器，浏览器在后续访问的时候带上这个Cookies，服务器就可以识别用户身份。基于唯一ID识别用户身份的机制叫做Session (
n.意为一段时间)。用户第一次访问服务器会自动获得一个Session ID，如果用户在一段时间内没有访问这个服务器，那么Session就会自动失效。下次访问服务器会分配一个新的Session
ID，将该用户看作是一个新用户。识别用户的名为Session机制是通过Cookies来实现的。

以用户登录为例，`HttpServletRequest`这个对象里面封装了用户请求信息，同时也提供了生成session
ID的方法。登录时判断用户名和密码，如果正确的话，对这个请求获取一个session，并将用户名称放入这个session中。之后在其他的servlet中，我们可以从
`HttpServletRequest`（封装了HTTP请求和Session）对象中获取到session。识别用户身份，进而继续处理用户的请求。用户登出的话，就是从
`HttpSession`中移除该用户的信息。

```java
HttpSession session = request.getSession();
session.setAttribute("user",userName);
```

### Cookies

是服务器识别用户身份，跟踪用会话的一串字符。服务器可以设置一个Cookies，发送给浏览器。浏览器下次就可以带着这个Cookies对服务器请求。服务器就可以识别用户身份。

### Filter

在一个复杂的应用程序中，有多个Servlet来处理不同的URI。有些功能的请求需要用户通过登录后才给放行，否则我们需要直接跳转到登录页面。那么这个判断登录的逻辑需要在这些Servlet中都写一遍。为了实现代码复用，我们把这些相同的功能从各个Servlet中抽离出来，在HTTP请求到达某些Servlet之前，先被一个中间商处理，然后在交给对应的Servlet。注意到这个分发器是不同的，这个中间商只对某些用户请求起作用，起到了对用户请求的预处理作用。因此我们把它叫做过滤器（Filter）。

对于那些需要用户登录才能操作的功能，我们把它放在更下一级的目录，对于这些URI的请求，都会先经过过滤器，然后才会分发到对应的Servlet。

多个过滤器会组成一个从前往后的链条，对于每个到达的请求会被链条上的过滤器依次处理。如果中间的某个过滤器内部在处理请求的时候，发现这个请求不符合预订的规则，调用了重定向，那么后续的过滤器将没有机会在处理该请求了。

### Listener
