---
order: 3
---

# Spring Web

## Controller 层

在 MVC 模式中，Controller 作为控制器，控制视图和模型之间的交流，使视图和模型分离。在 Web 应用中，Controller
层接收前端发来的请求，交由对应的服务层处理，然后返回响应结果。在整个架构中，Controller
可看作前端和后端交互的中介，由于前端发来的请求多种多样，后端负责处理这些请求的类及方法也不同，因此需要一个中间商接收前端请求，先进行简单处理，识别请求意图，然后交由对应方法处理。Controller
层因此具有承前启后的作用。

HTTP 请求分为请求行、请求头、请求体三部分。请求行携带请求方法、URL、HTTP 协议版本。请求头携带 HTTP
请求的必要信息，请求体是请求所携带的数据。每个请求向服务器请求的数据不同，因此请求行、请求头、请求体内容也不同。服务器要根据这些请求返回不同数据，首先需要分辨这些请求到底想请求什么。

Web 应用中 Controller 负责接收 HTTP 请求，Controller 层需要对请求进行分析处理，分析 HTTP 请求意图，然后交由 Service 层处理。在
Controller 层中，有两大任务：获取请求信息、返回响应数据。为处理好这两大任务，划分出以下几个步骤：

1. 首先，请求是多种多样的，单一 Controller 无法满足所有请求要求。先把请求分类，不同 URL 对应不同种类请求。URL
   有层级，可对请求种类进一步细分。因此设置不同的 Controller 类、不同的 Controller 方法处理不同种类请求，这时需要指定请求的映射规则
2. 其次，对于同一类请求，需要了解请求数据是什么，需要获取请求参数或获取请求体，进一步识别请求想要获取哪些数据
3. 还有一点，对于服务器中某些私密资源，不可能让任意请求都能获取到，因此需要识别 HTTP 请求身份，是否具有相关权限获取资源。需要获取请求头
4. 如果上述方法不能满足需要，可直接获取封装在对象中的 HTTP 请求全部信息，这叫做获取原生对象

### 指定请求映射规则

@RequestMapping 用于映射前端 HTTP 发来的 Request 请求，对于前端发来的不同请求，应指定不同的
Controller、不同的方法处理。RequestMapping 注解用于设置 HTTP 请求映射到对应 Controller 方法上的相关规则，如指定 Controller
方法处理的请求路径、请求方式、请求参数等配置。施加 @RequestMapping 注解上配置的参数会限制 HTTP 请求映射到该方法上的范围。

@RequestMapping 注解参数说明：

| 参数           | 值        | 说明                                                                                     |
|--------------|----------|----------------------------------------------------------------------------------------|
| name         | String   | 为该 RequestMapping 设置一个名字                                                               |
| value / path | String[] | 指定接收的 URI 路径。支持 Ant 样式路径匹配方法，yml 占位符如 `${path}`                                        |
| method       | enum[]   | 指定接收的请求方法。`public enum RequestMethod {GET, HEAD, POST, PATCH, DELETE, OPTIONS, TRACE}` |
| params       | String[] | 指定接受的请求参数。只有 HTTP 请求带有对应参数时，才会被该 Controller 处理，使用 `!` 表示不能具有该类请求                       |
| header       | String[] | 指定接收的请求头。具有某些请求头或某些请求头有特定值，才会被该 Controller 处理，使用 `!` 表示不能具有该类请求头                       |
| consumes     | String[] | 指定接收的请求内容类型 Content-Type                                                               |
| produces     | String[] | 指定从 HTTP 请求中发来的可接受响应的 Content-Type                                                     |

注：

1. @RequestMapping 可以使用在类上和方法上，方法上的 @RequestMapping 会继承类上已有的设置
2. Ant 样式路径匹配方法

| 路径 | 说明               | 实例                                   |
|----|------------------|--------------------------------------|
| ?  | 匹配任意单个字符，不包含 `/` | `/p?ttern` 匹配该文件夹下符合该规则的文件夹（不包含子文件夹） |
| *  | 匹配 0 或任意数量的字符    | `/*.jsp` 匹配当前文件夹下任何 JSP 文件（不包含子文件夹）  |
| ** | 匹配 0 或更多目录       | `/**/*.jsp` 匹配该文件夹及其子文件夹任何 JSP 文件    |

### 获取请求参数

#### 获取路径参数

@PathVariable 用于获取通过 URL 路径传递的请求参数，通常添加在 Controller 方法的参数中，Controller
方法所映射的路径中需要写明通过路径传递了哪些参数。@PathVariable 注解参数有两个：value（映射请求路径参数）、required（请求路径参数是否必须）。

```java
@RequestMapping("/user/{id}/{name}")
public String findUser(@PathVariable("id") Integer id,
                       @PathVariable("name") String name) {
    // TODO
}
```

#### 获取请求体中 JSON 格式参数

@RequestBody 用于获取请求体中的 JSON 数据，并将 JSON 数据转化为 Java 对象，需要 JSON 数据属性名和 Java 对象变量名一一对应，才能将数据传递到
Java 对象中，否则无法获取对应请求数据。

注意：使用 @RequestBody 获取请求体数据，需要请求头中的 Content-Type 值为 application/json，否则无法获取。

#### 获取 QueryParameter 格式参数

@RequestParam 用于获取 QueryParameter 格式参数。类似于 `URI?name1=value1&name2=value2` 格式在 URL 上传输的参数叫做
QueryParameter 格式参数。默认情况下，Controller 映射到的请求参数都是 QueryParameter 类型的参数，且需要请求中参数名和
Controller 方法中变量名一一对应，才能映射成功。

总结：通过 @RequestBody 和 @RequestParam 两个注解，可直接单独获取每个请求参数，也可将参数封装到自定义实体对象中，实体类中的成员变量要和请求参数名对应上，并提供对应的
set/get 方法。

#### @RequestBody 和 @RequestParam 注解的其他属性

| 属性           | 值       | 说明                   |
|--------------|---------|----------------------|
| required     | boolean | 请求参数是否必须传入           |
| defaultValue | String  | 如果没有传入对应请求参数，指定一个默认值 |

#### 参数类型转换

（待补充）

### 获取请求头和 Cookie

@RequestHeader 用于获取请求头信息，在注解中填写请求头名称可获取到对应请求头的值。

```java
@Controller
public class RequestResponseController {
    @RequestMapping("/getHeader")
    public String getHeader(@RequestHeader(value = "device-type") String deviceType) {
        System.out.println(deviceType);
        return "test";
    }
}
```

@CookieValue 用于获取 Cookie 信息，使用方法和 @RequestHeader 注解类似，在注解中填写 Cookie 的名称可获取到对应 Cookie 的值。

```java
@Controller
public class RequestResponseController {
    @RequestMapping("/getCookie")
    public String getCookie(@CookieValue("JSESSIONID") String sessionId) {
        System.out.println(sessionId);
        return "test";
    }
}
```

### 获取原生对象

使用 Servlet 时，Controller 获取 request 对象、response 对象、session 对象等。Spring MVC 帮助简化了对请求信息的处理，因此可通过一些注解直接获取想要的信息。但
Spring MVC 也提供了获取这些原生对象的方法，只需在方法上添加对应类型的参数即可。Spring MVC 会把需要的对象传给形参。这时需要使用
Servlet 的 API 来处理这些数据，会稍显繁琐。通常在需要设置响应头或进行文件传输时会获取原生对象，数据传输只需写入响应体。

```java
@Controller
public class RequestResponseController {
    @RequestMapping("/getReqAndRes")
    public String getReqAndRes(HttpServletRequest request, 
                              HttpServletResponse response, 
                              HttpSession session) {
        return "test";
    }
}
```

### 设置返回响应体

通过添加 @ResponseBody 注解可返回 JSON 格式的响应体，Spring MVC 会自动将 Java 对象转化为 JSON。

### 文件传输

#### 文件上传

HTTP 请求需要满足条件：

1. 请求方式为 POST
2. 请求头 Content-Type 为 multipart/form-data

Spring MVC 接收文件：需要 Controller 方法中的参数为 MultipartFile 类型。

#### 文件下载

Spring MVC 封装 HTTP 响应需要的条件：

1. 设置响应头的 Content-Type 属性为对应文件的 MIME 类型
2. 设置响应头的 Content-Disposition
3. 文件数据以二进制形式写入响应体中

## 拦截器

（待补充）

## 异常处理

Spring MVC 提供了注解 @ControllerAdvice 声明一个类为 Spring 管理的组件，可为特定 Controller 添加"通知"。这是 AOP 原理的实现，即将
@ControllerAdvice 中声明的方法织入到 Controller 中。

@ExceptionHandler 用于捕获 Controller 中抛出的异常，与 @ExceptionHandler 注解配合，可通过自定义拦截规则在 Controller
发生异常后进行拦截，在拦截后通过自定义方法继续拦截后处理，从而返回给前端自定义异常信息。

默认情况下，@ControllerAdvice 会在发生异常后拦截所有 Controller 然后进行处理。@RestControllerAdvice 会将返回值写入响应体中，相当于
@ControllerAdvice + @ResponseBody。总结来说可通过 @ControllerAdvice 和 @ExceptionHandler 实现全局异常处理。

```java
@ControllerAdvice
public class MyControllerAdvice {
    @ExceptionHandler({NullPointerException.class, ArithmeticException.class})
    @ResponseBody
    public Result handlerException(Exception ex) {
        Result result = new Result();
        result.setMsg(ex.getMessage());
        result.setCode(500);
        return result;
    }
}
```

## Spring MVC 执行流程

1. 用户发起请求被 DispatcherServlet 处理
2. DispatcherServlet 通过 HandlerMapping 根据 HTTP 请求内容查找能够处理该请求的 Handler（Controller）。HandlerMapping 处理
   HTTP 请求和处理方法之间的映射关系
3. HandlerMapping 返回一个能够处理请求的执行链给 DispatcherServlet，包含 Handler 方法和拦截器
4. HandlerAdapter 执行对应的 Handler 方法，把 HTTP 请求数据转换成合适类型，作为 Handler 的方法参数传入
5. Handler 方法执行完成后的返回值被放到响应体中，然后返回给 DispatcherServlet，再发送响应数据
