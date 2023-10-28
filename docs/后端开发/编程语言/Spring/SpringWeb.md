---
order: 3
---

# Spring Web

## 1、Controller 层

在 MVC 模式中，controller 作为控制器，控制视图和模型之间的交流，使视图和模型分离开。在 Web 应用中，也是类似的概念，控制层接受前端发来的请求，交由对应的服务层处理，然后返回响应结果。在整个架构中，我们大致上可以把 Controller 看做是前端和后端交互的中介，由于前端发来的请求多种多样，后端负责处理这些请求的类以及方法也都不同，因此我们需要一个中间商，接收前端发来的请求，先对其进行简单的处理，识别该请求的意图，然后交由对应的方法去处理。Controller 层因此有着承前启后的作用。

HTTP 请求分为请求行、请求头、请求体三部分。请求行中携带了请求方法、URL、HTTP 协议版本。请求头中携带了 HTTP 请求的一些必要信息，而请求体中是请求所携带的数据。每一个请求向服务器请求的数据都不太一样，因此请求行、请求头、请求体中的内容也不太一样。服务器要根据这些请求返回不同的数据，首先就是要分辨这些请求到底是想请求什么。

Web 应用中 Controller 负责接收 HTTP 请求，那么 Controller 层就需要对请求进行分析处理。分析 HTTP 请求的意图，然后交由 Service 层去处理。在 Controller 层中，我们有两大任务，获取请求信息、返回响应数据。为了处理好这两大任务，我们划分出以下几个步骤：

1. 首先，请求是多种多样的，单一的 Controller 无法满足所有请求的要求。我们先把请求分类，不同的 URL 对应着不同种类的请求。URL 是有层级的，我们可以对请求的种类再进一步细分。因此我们设置了不同的 Controller 类、不同的 Controller 方法、来处理不同种类的请求，这时候就需要**指定请求的映射规则**。
2. 其次，对于同一类的请求，我们就具体的了解请求的数据是什么，那么我们就需要**获取请求参数**或者**获取请求体**，来进一步识别请求是想要获取哪些数据。
3. 还有一点，对于服务器中某些私密的资源，我们不可能让任意的请求都能获取到，因此需要识别 HTTP 请求的身份，是否具有相关的权限去获取资源。那么我们就需要**获取请求头**。
4. 如果上面的方法不能够满足我们需要的话，我们可以直接获取封装在一个对象中的 HTTP 请求的全部信息，这叫做**获取原生对象**。
5.

### 1）指定请求映射规则

@RequestMapping 用于映射前端 HTTP 发来的 Request 请求，对于前端发来的不同请求，我们应该指定不同的 Controller、不同的方法来处理。RequestMapping 注解就是让我们设置 HTTP 请求映射到对应 controller 方法上的相关规则，比如说指定一个 controller 方法处理的请求路径、请求方式、请求参数等等一系列配置。施加@RequestMapping 注解上配置的参数会限制 HTTP 请求映射到该方法上的范围。

@RequestMapping 注解参数说明

| 参数         | 值       | 说明                                                         |
| ------------ | -------- | ------------------------------------------------------------ |
| name         | String   | 为该 RequestMapping 设置一个名字                             |
| value / path | String[] | 指定接收的 URI 路径。支持 Ant 样式路径匹配方法，yml 占位符如.`${path}` |
| method       | emum[]   | 指定接收的请求方法。`public enum RequestMethod {GET,HEAD,POST,PATCH,DELETE,OPTIONS,TRACE}` |
| params       | String[] | 指定接受的请求参数。只有 HTTP 请求带有对应的参数时，才会被该 Controller 处理，使用 `!`表示不能具有该类请求。 |
| header       | String[] | 指定接收的请求头。具有某些请求头或者某些请求头有特定的值，才会被该 Controller 处理，使用 `!`表示不能具有该类请求头。 |
| consumes     | String[] | 指定接收的请求内容类型 Content-Type                          |
| produces     | String[] | 指定从 HTTP 请求中发来的可接受响应的 Content-Type            |

注：

1、注意到@RequestMapping 可以使用在类上和方法上，在方法上的@RequestMapping 会继承类上已有的设置。

2、Ant 样式路径匹配方法

| 路径 | 说明                         | 实例                                                         |
| ---- | ---------------------------- | ------------------------------------------------------------ |
| ?    | 匹配任意单个字符，不包含 `/` | `/p?ttern`匹配该文件夹下符合该规则的的文件夹（不包含子文件夹） |
| \*   | 匹配 0 或者任意数量的字符    | `/*.jsp`匹配当前文件夹下任何 JSP 文件（不包含子文件夹）      |
| \*\* | 匹配 0 或者更多的目录        | `/**/*.jsp`匹配该文件夹及其子文件夹任何 JSP 文件             |

### 2）获取请求参数

#### 获取路径参数

@PathVariable 用来获取通过 URL 路径传递的请求参数，通常添加在 Controller 方法的参数中，Controller 方法所映射的路径中需要写明通过路径传递了哪些参数。@PathVariable 注解参数有两个，分别是 value：映射请求路径参数，required：请求路径参数是否必须

```java
@RequestMapping("/user/{id}/{name}")
public String findUser(@PathVariable("id")  Integer id,
                       @PathVariable("name") String name){
    // TODO
}
```

#### 获取请求体中 JSON 格式参数

@RequestsBody 用来获取请求体中的 JSON 数据，并将 JSON 数据转化为 JAVA 对象，需要 JSON 数据属性名和 JAVA 对象变量名一一对应，才回将数据传递到 Java 对象中，否则无法获取对应的请求数据。

注意：使用@RequestsBody 获取请求体数据，需要请求头中的 Content-Type 值为 application/json 否则会无法获取。

#### 获取 QueryParameter 格式参数

@RequestParam 用于获取 QueryParameter 格式参数。类似于 `URI?name1=value1&name2=value2`格式在 URL 上传输的参数叫做 QueryParameter 格式参数，默认情况下，Controller 映射到的请求参数都是 QueryParameter 类型的参数，且需要请求中参数名和 Controller 方法中变量名一一对应，才能映射成功。

**总结：**通过 `@RequestsBody`和 `@RequestParam`两个注解，我们可以直接单独获取每一个请求参数，也可以将参数封装到自定义实体对象中，实体类中的成员变量要和请求参数名对应上。并且要提供对应的 set/get 方法。

#### `@RequestsBody`和 `@RequestParam`注解的其他属性

| 属性         | 值      | 说明                                     |
| ------------ | ------- | ---------------------------------------- |
| required     | boolean | 请求参数是否必须传入                     |
| defaultValue | String  | 如果没有传入对应请求参数，指定一个默认值 |

#### 参数类型转换

// TODO

### 3）获取请求头和 cookie

@RequestsHeader 用于获取请求头信息，在注解中填写请求头名称我们就可以获取到对应请求头的值

```java
@Controller
public class RequestResponseController {
    @RequestMapping("/getHeader")
    public String getHeader(@RequestHeader(value = "device-type") String deviceType){
        System.out.println(deviceType);
        return "test";
    }
}
```

@CookieValue 用于获取 cookie 信息，使用方法和@RequestsHeader 注解类似，在注解中填写 cookie 的名称我们就可以获取到对应 cookie 的值

```java
@Controller
public class RequestResponseController {
    @RequestMapping("/getCookie")
    public String getCookie(@CookieValue("JSESSIONID") String sessionId){
        System.out.println(sessionId);
        return "test";
    }
}
```

### 4）获取原生对象

我们之前使用 servlet 的时候，Controller 获取的是 request 对象，response，session 对象等。SpringMVC 帮助我们简化了对请求信息的处理，因此我们可以通过一些注解直接获取到我们想要的信息。但是 SpringMVC 中也提供了获取这些原生对象的方法，只需要在方法上添加对应类型的参数就行。SpringMVC 会把我们需要的对象传给我们的形参。不过这时候我们就需要使用 servlet 的 API 来处理这些数据，会稍显繁琐和麻烦。通常在我们需要设置响应头或者进行文件传输时会获取原生的对象，数据传输只需要写入响应体就可以了。

```java
@Controller
public class RequestResponseController {
    @RequestMapping("/getReqAndRes")
    public String getReqAndRes(HttpServletRequest request, HttpServletResponse response, HttpSession session){
        return "test";
    }
}
```

### 5）设置返回响应体

我们通过添加@ResponseBody 注解就可以返回 JSON 格式的响应体，springMVC 会为我们自动将 Java 对象转化为 JSON

### 6）文件传输

#### 文件上传

HTTP 请求需要满足条件：1、请求方式为 POST。2、请求头 Content-Type 为 multipart/form-data

SpringMVC 接收文件：需要 Controller 方法中的参数为 MutipartFile 类型的。该类型有以下几种常见的方法

| 方法 | 说明 |
| ---- | ---- |
|      |      |

#### 文件下载

SpringMVC 封装 HTTP 响应需要的条件：1、设置响应头的 Content-Type 属性为对应文件的 MIME 类型。2、设置响应头的 Content-Disposition。3、文件数据以二进制形式写入响应体中。

## 2、拦截器

## 3、异常处理

SpringMVC 为我们提供了注解@ControllerAdvice 声明一个类为 spring 管理的一个组件，可以为特定的 Controller 添加“通知”。是 AOP 原理的实现，也就是说将@ControllerAdvice 中声明的方法织入到 Controller 中。

@ExceptionHandler 用于捕获 Controller 中抛出的异常，与@ExceptionHandler 注解配合，我们可以通过自定义的拦截规则在 Controller 发生异常之后进行拦截，在拦截之后，我们转而通过自定义的方法来继续拦截后的处理，从而返回给前端自定义的异常信息。

默认情况下，@ControllerAdvice 会在发生异常后拦截所有的 Controller 然后进行处理。@RestControllerAdvice 会将返回值写入响应体中，相当于@ControllerAdvice + @ResponseBody 。总结来说可以通过@ControllerAdvice 和@ExceptionHandler 实现全局的异常处理。

```java
@ControllerAdvice
public class MyControllerAdvice {
    @ExceptionHandler({NullPointerException.class,ArithmeticException.class})
    @ResponseBody
    public Result handlerException(Exception ex){
        Result result = new Result();
        result.setMsg(ex.getMessage());
        result.setCode(500);
        return result;
    }
}

```

## 4、SpringMVC 执行流程

1. 用户发起请求被 DispatchServlet 所处理
2. DispatchServlet 通过 HandlerMapping 根据 HTTP 请求内容查找能够处理这个请求的 Handler（Controller）。HandlerMapping 就是用来处理 HTTP 请求和处理方法之间的映射关系
3. HandlerMapping 返回一个能够处理请求的执行链给 DispatchServlet，包含了 Handler 方法和拦截器
4. HandlerAdapter 执行对应的 Handler 方法，把 HTTP 请求数据转换成合适的类型，作为 Handler 的方法参数传入
5. Handler 方法执行完成之后的返回值被放到响应体中，然后返回给 DispatchServlet，然后发送响应数据
