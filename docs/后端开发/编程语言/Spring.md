# Spring FrameWork

## 1、`IoC`容器

### 1）容器的概念

对于一个可以存放数据的具体数据结构的实现，都可以叫做容器。spring 的容器是为某些特定组件对象提供必要支持的一个软件环境。这些特定的组件对象就是 Spring
Bean，他提供了一些底层服务如对象的配置、对象整个生命流程的管理，让容器所承载的对象不必在考虑这些问题。Tomcat 就是一个 Servlet 容器，底层实现了 TCP 连接，解析 HTTP 协议等非常复杂的服务。我们自己就无需在组件中编写这些复杂的逻辑。IoC 容器它可以管理所有轻量级的 JavaBean 组件，提供的底层服务包括组件的生命周期管理、配置和组装服务、AOP 支持，以及建立在 AOP 基础上的声明式事务服务等。

如果把一个 Bean 对象交给 Spring 容器管理。这个对象会被拆解存放到 Bean 的定义中。然后在由 Spring 统一装配，在装配中包括 Bean 的初始化、属性填充。对于 Spring 容器，我们需要一种可以用于存放对象、可以通过名称索引查找对象的数据结构，那么我们选择 HashMap。Spring 容器的实现需要对象的定义、注册、获取三个基本步骤。

- 定义：定义存放在 Spring 中的对象，这里我们把这种对象称为 bean 对象。因此这个定义的过程称为 BeanDefinition。我们设计他是一个类。里面包含 singleton、prototype、BeanClassName 等属性。
- 注册：将 Bean 对象注册到 Spring 容器中，或者称为将对象放入 HaspMap 中以便我们后续的获取。Key 为 Bean 对象的名称，Value 为对象本身。
- 获取：通过 Bean 对象的名称可以获取该对象。

### 2）容器的设计

Spring 中 Bean 对象的创建是交给容器本身来完成的，而不是我们在调用是传入一个已经实例化的对象。对于同一个类型的对象，有时我们需要一个有时需要多个，那么我们需要重点考虑的是单例对象。

### 2）控制反转 IoC

`IoC`意为控制反转（Inversion of Control），对程序中对象的创建、配置这样的控制权由应用程序转移到了 `IoC`
容器，那对于某个具体的实例对象它的所有组件对象不再由应用程序自己创建和配置，而是通过 `IoC`容器负责。这样应用程序能够直接使用已经创建并配置好的组件。

在设计上 `IoC`
容器是一个无侵入的容器，应用程序的组件无需实现 Spring 的特定接口，那么这些组件即可以在 spring 容器中运行，又能够自己编写代码组装他所需要的对象。还有就是在测试的时候，也就不需要实现接口，不依赖与 Spring 容器，可单独测试。

### 3）依赖注入

这些组件需要通过注入机制来装入到实例对象中，供实例对象使用。依赖注入的方式可以有两种，一种是通过 `setXXX()`方法注入，另一种是通过构造方法实现。Spring 的 IoC 容器同时支持属性注入和构造方法注入，并允许混合使用。

因为 `IoC`容器需要负责实例化所有组件对象，所以需要告诉容器如何创建组件对象，以及各个组件对象之间的依赖关系，即装配方式。在 Spring 可以通过两种方式实现，一种是 XML 配置文件，另一种是通过注解。

### 4）组件装配

#### I 通过 XML 装配组件

我们需要自己将组件之间的依赖关系描述出来，然后交给容器来创建并装配。

**第一步 编写配置文件 application.xml**

我们需要编写一个特定的名叫 application 的配置文件 `application.xml`告诉 Spring 容器应该如何创建、并按顺序正确的注入到相应的组件中。Bean 表示这是一个 Java Bean 或者说是一个组件。id 唯一标识了一个 Java Bean，class 提供了文件路径。每个 Java Bean 内部可以有一个或多个需要注入的属性，以 property 标签表示。而这些属性也是一个 Java Bean，name 表示在这个组件内部这个需要注入的属性的名称是什么。ref 表示这个需要注入的属性所指向的 Java Bean 的 id。这些 Java Bean 在配置文件的相对位置并不重要，但是每个组件中要注入的属性需要写全，不写全的画 spring 会漏掉注入该属性。如果注入的不是 Java Bean 那么将 ref 改为 value。

总结来说，Java Bean 通过引用注入，数据类型通过 value 注入。

```xml
<bean id="userService" class="com.itranswarp.learnjava.service.UserService">
    <property name="mailService" ref="mailService" />			<!--引用注入-->
    <property name="username" value="root" />				    <!--值注入-->
    <property name="password" value="password" />
</bean>
```

**第二步 在代码中加载配置文件**

我们需要创建一个 `Spring IoC`容器的实例，然后加载配置文件。接下来我们就可以从 Spring 容器中取出组件并使用它。Spring 容器命名为应用程序上下文，就是 `ApplicationContext`，它是一个接口，用来加载配置文件，有很多实现类。通过 xml 加载需要 `ClassPathXmlApplicationContext`实现类来帮我们自动从项目路径下查找指定的配置文件，参数为配置文件名。通过注解加载需要 `AnnotationConfigApplicationContext`实现类，参数为配置类名称，必须传入一个标注了 `@Configuration`的类名。。

#### II 通过注解装配组件

见组件详解

## 2、AOP

在实际开发中有很多功能是许多组件通用的，但又是非核心的业务逻辑。让框架把一些很多个不同的组件之间通用的非核心的业务逻辑通过某种方法，织入到组件中。那么 AOP 要把切面即一些非核心、但又必要的逻辑织入核心逻辑中，我们在调用某个业务方法时，spring 会对该方法进行拦截，并在拦截前后进行安全检查、日志、事务等处理。从而完成了整个业务流程。有 3 种方式实现。

- 编译期，由编译器把切面（非核心的逻辑）编译进字节码。
- 类加载器：当目标被装载到 JVM 时，通过一个特殊的类加载器，对目标类的字节码重新增强
- 运行期：通过动态代理实现运行期动态织入。

Spring 的 AOP 实现就是基于 JVM 的动态代理，通过 AOP 技术，可以让我们把一些常用的功能如权限检查、日志、事务等，从每个业务方法中剥离出来。

我们使用 AOP 非常简单，一共需要三步：

1. 定义切入方法，并在方法上通过 AspectJ 的注解告诉 Spring 应该在何处调用此方法；
2. 在需要切入方法的地方标记 `@Component`和 `@Aspect`；
3. 在 `@Configuration`类上标注 `@EnableAspectJAutoProxy`。

我们还可以通过自定义注解来切入功能。在那些需要切入这种常用的功能的方法头上，标记一个自定义注解，而在切入方法（常用的功能逻辑所在的方法）的 AOP 注解参数中填入该注解的名称，参数格式为 `"@annotation(your_annotation_name)"`，那么只要标注了你自定义注解的地方，spring 都会把切入方法切入到里面。

# 二、Spring Web

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

# 三、Spring Boot

# 四、Spring Security

## 1、概述

### 认证与授权

一般的 Web 应用都需要对用户进行认证和授权两个操作。

认证：Authentication 验证当前访问系统的用户是不是本系统的用户，并且需要确认具体是哪一个用户。

授权： Authorization 对经过认证后的用户判断它是否有权限进行某个操作

### 一般流程

1. 当用户登录时，前端将用户输入的用户名密码传输到后台，后台使用一个类对象将其封装起来，在 Spring Security 中使用的是 `UsernamePasswordAuthenticationToken`类
2. 程序需要负责验证前端传来的这个对象，验证方法就是调用 service 服务——根据用户名从数据库中取出用户信息到实体类的实例中，然后比较两者密码，如果密码正确就成功登录。登录成功的同时把包含着用户信息如用户名、密码、用户具有的权限等信息的一个对象放到类似于 Session 的对象中（SecurityContextHolder）
3. 用户访问一个资源的时候，首先要判断该资源是否是限制访问的资源，如果是的化，需要判断当前用户是否登录，没有登录就跳转到登录页面。
4. 如果用户已经登陆，访问某一个限制访问的资源，程序要根据用户请求的地址（url）从数据库中取出该资源对应的所有可以访问该资源的角色，与当前用户所对应的角色一一对比，判断用户是否可以访问。

![[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-LKooVd3p-1648311184478)(C:\Users\30666\AppData\Roaming\Typora\typora-user-images\image-20220326160405082.png?lastModify=1648282919)]](https://img-blog.csdnimg.cn/993321379a3843f2bb0cb8df913acb2a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5LiD5a-75YyX6YeM,size_20,color_FFFFFF,t_70,g_se,x_16)

用户提交的用户名和密码首先经过了一个认证过滤器，在这里将请求信息封装成一个类——Authentication 类，这个类会存放用户信息，并且之后会存放用户的认证结果。然后调用认证方法到下一个地方进行认证。（也就是说判断是否是当前系统的用户）这里会通过一个认证管理器，认证管理器委托数据库访问认证提供者进行认证，这个提供者主要负责认证逻辑处理工作，首先从 `UserDetailsService`用户详情服务商中调用 `loadUserByUsername()`获取用户信息，会返回的用户详情信息类实例 `UserDetails`，这个提供者通过密码加密器对比前端传来的密码和数据库中存在的密码是否一致。然后根据比对结果填充封装后的请求信息类。随后，过滤器便收到了认证的结果，过滤器将结果保存在服务器的一个叫做 `SecurityContextHolder`的地方。通过这样一个方法 `SecurityContextHolder.getContext().setAuthentication( authentication )`将认证结果保存。

部分方法说明：

1. `UsernamePasswordAuthenticationFilter`是处理用户名和密码认证方式的一个过滤器，将请求信息封装为 Authentication 类
2. `AuthenticationManager`是一个接口，定义了认证的方法，也是认证的入口，在这里他是用来管理各种认证方法的。有一个实现类 ProviderManager 里面会有一列表的认证方式。我们在自己写的登录认证 Service 中需要注入这样的接口，调用他的认证方法进行认证。会返回认证信息 Authentication。我们可以对返回的认证信息进行后续的操作，如获取其中的用户信息构造 jwtToken，然后将其存放在 redis 中
3. UserDetailsService 接口：在这里我们需要指定相应的从数据库查询用户信息的方法。我们需要自己实现这个接口，注入一个 DAO 然后将其从数据库中查询到的用户信息转换成 Security 中要求的 UserDetails 对象即可。
4. UserDetails 接口：这个接口存放了用户信息，以及与用户认证和授权有关的方法，Security 会调用这个接口的实现类来判断账户的认证授权情况。

## 2、认证

## 3、授权

## 4、自定义失败处理

## 5、跨域

## 6、自定义权限校验方法

## 7、CSRF

## 8、认证处理器

# 五、Mybatis

## 1、介绍

我们把 mybatis 的功能架构分为了三层：

1. 接口层：提供给程序员调用的接口 API，程序员通过这些 API 来操作数据库，接口层收到调用请求就会转而调用数据处理层来完成具体的数据处理。
2. 数据处理层：负责对具体的从参数映射、SQL 语句的解析、执行、以及执行结果的映射处理等，主要的作用就是根据调用的请求完成一次数据库的操作。
3. 基础支撑层：负责最基础的功能支撑，包括连接池管理、事务管理、配置的加载、缓存处理。这些都是进行数据库查询通用的东西，我们把他从数据库操作中抽取出来作为最基本的组件，为上层的数据处理层提供最基础的支撑。

最后是引导层，引导层不参与 SQL 语句的处理，它只是为了在应用程序中配置 mybatis 的各种功能如何去运行。

## 2、总体框架设计

### 1）接口层

接口层是 mybatis 提供给程序员调用的应用程序接口，我们通过接口层就可以方便的调用数据库获取数据。目前 mybatis 提供两种方式与数据库进行交互。

- 创建 SqlSession 对象。第一种是创建一个 SQL 语句会话对象（SqlSession）使用该对象完成对数据库的交互，该对象内部维护了与数据库的连接，提供了与数据库进行增删改查操作的方法。
- 使用 Mapper 接口。第二种是通过接口调用的方式，mybatis 中应用程序的某个对象与数据库某个表连接的桥梁是通过 mapper 映射实现的，配置文件中的每一个 mapper 结点都对应着一个 mapper 接口。接口中的每一个方法对应这配置文件中每一条 SQL 语句。我们在配置好以后，mybatis 会根据相应的接口方法通过动态代理生成一个 mapper 实例，我们在调用 mapper 接口的某一个方法的时候，mybatis 会根据这个方法的名称还有参数类型，确定 SQL 语句，实现对数据库的操作。

### 2）数据处理层

我们在 Java 中调用这些对数据库增删改查方法的时候，会传入一些参数，这些参数可能是具体要存的数据，或者是其他的东西，mybatis 的数据处理层所要实现的功能就是从这里展开的，主要完成两个功能

- 一是通过传入的参数构建动态的 SQL 语句
- 二是执行对应的 SQL 语句并封装查询结果映射到 Java 对象中。

### 3）框架支撑层

框架支撑层，负责数据库查询中一些通用的东西，主要有以下几点内容

- 事务管理机制：
- 连接池管理机制：
- 缓存机制：为了减少数据库的压力，提高数据利用率，mybatis 会对一些查询结果缓存到本地中，在一定的时间间隔内多次的同一查询，mybatis 会直接返回缓存的内容，而不会再去数据库中查找。
- SQL 语句的配置方式：Java 程序中 SQL 语句配置方式有两种，一种是通过 XML 来配置，另一种是在 mapper 接口上使用注解来配置。这个功能就是为了识别并处理两种不同的配置方式而存在的。

## 3、实现功能的层次结构

1. 我们使用 mybatis 查询数据库，首先就是要创建一个 SQL 会话对象，也就是 SqlSession，创建完成之后，就开启了一个与数据库的连接会话，我们可以通过这个对象，来执行 SQL 语句、提交或者回滚事务。
2. 但是实际上出于分离职责、防止一个对象身兼太多职责，SqlSession 只是执行数据库查询的第一层对象，它会紧接着调用 Executor 对象，这个对象会负责 SQL 动态语句的生成，对查询出来的结果进行缓存，对这些结果进行维护，定期删除等。
3. 其次是语句处理器对象，由于 Java 程序在底层与数据库的交互是通过 JDBC 实现的，mybatis 是在 JDBC 的基础上做出了进一步的封装。因此语句处理器对象 StatementHandler 主要负责与 JDBC 语句之间的交互。设置语句参数，将返回的结果映射到 Java 对象。
4. 接下来就是 JDBC 层，是真正负责查询数据库的东西。

### 4、mybatis 初始化

我们如果想要在自己的程序中引入一个插件或者是框架，单单只把软件包导入进来是没有作用的，我们要在程序中使用它就需要进行一系列的配置，就比如 Java 的那些内置工具，他就在那但我们不能直接使用，我们调用的时候新建该对象，需要传入一些参数。类比到框架的初始化及配置上，就是这个道理。

mybatis 初始化的方式主要有两种：一种是通过 XML 配置。第二种是基于 Java 的 API

### 1）XML 配置初始化

### 2）Java 的 API 初始化

# 六、Mybatis Plus

## 1、在项目中引入 Mybatis Plus

- 第一步，添加相应依赖
- 第二步，需要在 Spring Boot 启动类中添加 `@MapperScan` 注解，扫描 Mapper 文件夹
- 第三步，编写 Mapper 包下的接口，继承 Mybatis Plus 提供的 `BaseMapper<T>`

默认情况下，Mybatis Plus 实体类有如下的映射关系：

| 映射关系                       | 说明                                                      |
| ------------------------------ | --------------------------------------------------------- |
| 下划线映射为实体类的驼峰式命名 | 表名：st_user -> stUser 类。字段名：real_name -> realName |
| 数据表主键名为 id              | 插入数据时会自增，不需要我们进行指定                      |
| 字段名与实体类属性一一对应     |                                                           |

对应注解

| 注解        | 说明                         |
| ----------- | ---------------------------- |
| @TableName  | 表名注解，标识实体类对应的表 |
| @TableId    | 主键注解，用于实体类主键字段 |
| @TableField | 非主键的字段注解             |

常见配置

```yaml
mybatis-plus:
  # 该配置项可以在编写 mapper.xml 文件时省略 resultType 的全类名，直接使用相应包下的类名即可
  type-aliases-package: com.itheima.mp.domain.po
  global-config:
    db-config:
      logic-delete-field: deleted #配置逻辑删除字段
  configuration:
    # 配置枚举类与pojo类相关属性及数据库相应数据之间的映射关系
    default-enum-type-handler: com.baomidou.mybatisplus.core.handlers.MybatisEnumTypeHandler
```

## 2、核心功能

### 2.1. 条件构造器

#### 2.1.1. QueryWrapper

无论是修改、删除、查询，都可以使用 QueryWrapper 来构建查询条件。

查询：查询出名字中带 `o`的，存款大于等于 1000 元的人，代码如下：

```java
@Test
void testQueryWrapper() {
    // 1.构建查询条件 where name like "%o%" AND balance >= 1000
    QueryWrapper<User> wrapper = new QueryWrapper<User>()
            .select("id", "username", "info", "balance")
            .like("username", "o")
            .ge("balance", 1000);
    // 2.查询数据
    List<User> users = userMapper.selectList(wrapper);
    users.forEach(System.out::println);
}
```

更新：更新用户名为 jack 的用户的余额为 2000，代码如下：

```java
@Test
void testUpdateByQueryWrapper() {
    // 1.构建查询条件 where name = "Jack"
    QueryWrapper<User> wrapper = new QueryWrapper<User>().eq("username", "Jack");
    // 2.更新数据，user中非null字段都会作为set语句
    User user = new User();
    user.setBalance(2000);
    userMapper.update(user, wrapper);
}
```

#### 2.1.2. UpdateWrapper

基于 BaseMapper 中的 update 方法更新时只能直接赋值，对于一些复杂的需求就难以实现。
例如：更新 id 为 `1,2,4`的用户的余额，扣 200，对应的 SQL 应该是：

```sql
UPDATE user SET balance = balance - 200 WHERE id in (1, 2, 4)
```

SET 的赋值结果是基于字段现有值的，这个时候就要利用 UpdateWrapper 中的 setSql 功能了：

```java
@Test
void testUpdateWrapper() {
    List<Long> ids = List.of(1L, 2L, 4L);
    // 1.生成SQL
    UpdateWrapper<User> wrapper = new UpdateWrapper<User>()
            .setSql("balance = balance - 200") // SET balance = balance - 200
            .in("id", ids); // WHERE id in (1, 2, 4)
        // 2.更新，注意第一个参数可以给null，也就是不填更新字段和数据，
    // 而是基于UpdateWrapper中的setSQL来更新
    userMapper.update(null, wrapper);
}
```

#### 2.1.3. LambdaQueryWrapper & LambdaUpdateWrapper

无论是 QueryWrapper 还是 UpdateWrapper 在构造条件的时候都需要写死字段名称，会出现字符串 `魔法值`。这在编程规范中显然是不推荐的。
那怎么样才能不写字段名，又能知道字段名呢？

其中一种办法是基于变量的 `gettter`方法结合反射技术。因此我们只要将条件对应的字段的 `getter`方法传递给 MybatisPlus，它就能计算出对应的变量名了。而传递方法可以使用 JDK8 中的 `方法引用`和 `Lambda`表达式。
因此 MybatisPlus 又提供了一套基于 Lambda 的 Wrapper，包含两个：

- LambdaQueryWrapper
- LambdaUpdateWrapper

分别对应 QueryWrapper 和 UpdateWrapper

使用方法如下：

```java
@Test
void testLambdaQueryWrapper() {
    // 1.构建条件 WHERE username LIKE "%o%" AND balance >= 1000
    QueryWrapper<User> wrapper = new QueryWrapper<>();
    wrapper.lambda()
            .select(User::getId, User::getUsername, User::getInfo, User::getBalance)
            .like(User::getUsername, "o")
            .ge(User::getBalance, 1000);
    // 2.查询
    List<User> users = userMapper.selectList(wrapper);
    users.forEach(System.out::println);
}
```

### 2.2. 自定义 sql

SQL 语句最好都维护在持久层，而不是业务层。就上述 UdateWrapper 案例来说，由于条件是 in 语句，只能将 SQL 写在 Mapper.xml 文件，利用 foreach 来生成动态 SQL。这就比较麻烦，所以，MybatisPlus 提供了自定义 SQL 功能，可以利用 Wrapper 生成查询条件，再结合 Mapper.xml 编写 SQL。

#### 2.2.1. 基本用法

对于上述 UpdateWrapper 案例，可以改为如下写法：

```java
// 模拟业务层（service 层）调用mapper
@Test
void testCustomWrapper() {
    // 1.准备自定义查询条件
    List<Long> ids = List.of(1L, 2L, 4L);
    QueryWrapper<User> wrapper = new QueryWrapper<User>().in("id", ids);

    // 2.调用mapper的自定义方法，直接传递Wrapper
    userMapper.deductBalanceByIds(200, wrapper);
}
```

然后在 mapper 中自定义 sql，其中 @Param("ew") 为必写项，并且只能写 `ew`

```java
public interface UserMapper extends BaseMapper<User> {
    @Select("UPDATE user SET balance = balance - #{money} ${ew.customSqlSegment}")
    void deductBalanceByIds(@Param("money") int money, @Param("ew") QueryWrapper<User> wrapper);
}
```

#### 2.2.2. 多表关联

理论上来讲 MyBatisPlus 是不支持多表查询的，不过我们可以利用 Wrapper 中自定义条件结合自定义 SQL 来实现多表查询的效果。(个人感觉还是写 mapper 文件比较方便)

```java
// 模拟业务层（service 层）调用mapper
@Test
void testCustomJoinWrapper() {
    // 1.准备自定义查询条件
    QueryWrapper<User> wrapper = new QueryWrapper<User>()
            .in("u.id", List.of(1L, 2L, 4L))
            .eq("a.city", "北京");

    // 2.调用mapper的自定义方法
    List<User> users = userMapper.queryUserByWrapper(wrapper);

    users.forEach(System.out::println);
}
```

在 UserMapper 中自定义方法：

```java
@Select("SELECT u.* FROM user u INNER JOIN address a ON u.id = a.user_id ${ew.customSqlSegment}")
List<User> queryUserByWrapper(@Param("ew")QueryWrapper<User> wrapper);
```

#### 2.2.3 练习

用户表结构:

```sql
CREATE TABLE `t_user`  (
  `uid` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `user_name` varchar(30) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '姓名',
  `age` int(11) NULL DEFAULT NULL COMMENT '年龄',
  `email` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '邮箱',
  `is_deleted` int(255) NULL DEFAULT 0,
  `sex` int(255) UNSIGNED ZEROFILL NULL DEFAULT NULL,
  PRIMARY KEY (`uid`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 41 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;
```

练习:

```java
//查询用户名包含a，年龄在20到30之间，邮箱不为空的用户信息
//SELECT uid AS id,user_name AS name,age,email,is_deleted FROM t_user WHERE is_deleted=0 AND (user_name LIKE ? AND age BETWEEN ? AND ? AND email IS NOT NULL)
@Test
public void test01(){
    QueryWrapper<User> queryWrapper = new QueryWrapper<>();
    queryWrapper.like("user_name", "a").between("age", 20, 30).isNotNull("email");
    List<User> list = userMapper.selectList(queryWrapper);
    list.forEach(System.out::println);
}

//查询用户信息，按照年龄的降序排序，若年龄相同，则按照id升序排序
//SELECT uid AS id,user_name AS name,age,email,is_deleted FROM t_user WHERE is_deleted=0 ORDER BY age DESC,uid ASC
@Test
public void test02(){
    QueryWrapper<User> queryWrapper = new QueryWrapper<>();
    queryWrapper.orderByDesc("age").orderByAsc("uid");
    List<User> list = userMapper.selectList(queryWrapper);
    list.forEach(System.out::println);
}

//删除邮箱地址为null的用户信息
@Test
public void test03(){
    QueryWrapper<User> queryWrapper = new QueryWrapper<>();
    queryWrapper.isNull("email");
    int result = userMapper.delete(queryWrapper);
    System.out.println("result="+result);
}

//将（年龄大于20并且用户名中包含有a）或邮箱为null的用户信息修改
//UPDATE t_user SET age=?, email=? WHERE is_deleted=0 AND (user_name LIKE ? AND age > ? OR email IS NULL)
@Test
public void test04(){
    QueryWrapper<User> queryWrapper = new QueryWrapper<>();
    queryWrapper
            .like("user_name", "a")
            .gt("age", 20)
            .or()
            .isNull("email");
    User user = new User();
    user.setAge(18);
    user.setEmail("user@atguigu.com");
    int result = userMapper.update(user, queryWrapper);
    System.out.println("result="+result);
}

//将用户名中包含有a并且（年龄大于20或邮箱为null）的用户信息修改
//lambda表达式内的逻辑优先运算
//UPDATE t_user SET age=?, email=? WHERE is_deleted=0 AND (user_name LIKE ? AND (age > ? OR email IS NULL))
@Test
public void test05(){
    QueryWrapper<User> queryWrapper = new QueryWrapper<>();
    queryWrapper
            .like("user_name", "a")
            .and(i->i.gt("age", 20).or().isNull("email"));
    User user = new User();
    user.setAge(21);
    user.setEmail("user@wl.com");
    int result = userMapper.update(user,queryWrapper);
    System.out.println("result="+result);
}

//查询用户信息的username、age和email字段
//SELECT user_name,age,email FROM t_user WHERE is_deleted=0
@Test
public void test06(){
    QueryWrapper<User> queryWrapper = new QueryWrapper<>();
    queryWrapper.select("user_name","age","email");
    //selectMaps()返回Map集合列表，通常配合select()使用，避免User对象中没有被查询到的列值为null
    List<Map<String, Object>> maps= userMapper.selectMaps(queryWrapper);
    maps.forEach(System.out::println);
}

//子查询案例
//查询id小于等于30的用户信息
@Test
public void test07(){
    QueryWrapper<User> queryWrapper = new QueryWrapper<>();
    queryWrapper.inSql("uid", "select uid from t_user where uid <= 30")
    .select("uid","user_name","email");
//        List<User> list = userMapper.selectList(queryWrapper);
//        list.forEach(System.out::println);
    List<Map<String,Object>> maps = userMapper.selectMaps(queryWrapper);
    maps.forEach(System.out::println);
}

//将用户名中包含有a并且（年龄大于20或邮箱为null）的用户信息修改
@Test
public void test08(){
    UpdateWrapper<User> updateWrapper = new UpdateWrapper<>();
    updateWrapper
            .like("user_name", "a")
            .and(i -> i.gt("age", 20).or().isNull("email"));
    updateWrapper.set("user_name", "小黑").set("email", "xiaohei@atguigu.com");
    int result = userMapper.update(null, updateWrapper);
    System.out.println("result="+result);
}

//定义查询条件，有可能为null（用户未输入或未选择）
@Test
public void test08Condition(){
    String username = null;
    Integer ageBegin = 10;
    Integer ageEnd = 24;
    QueryWrapper<User> queryWrapper = new QueryWrapper();
    //SELECT uid AS id,user_name AS name,age,email,is_deleted FROM t_user WHERE is_deleted=0 AND (age >= ? AND age <= ?)
    queryWrapper.like(StringUtils.isNotBlank(username),"user_name", "a")
            .ge(ageBegin != null,"age",ageBegin)
            .le(ageBegin != null,"age", ageEnd);
    List<User> list = userMapper.selectList(queryWrapper);
    list.forEach(System.out::println);
}

//LambdaQueryWrapper
//定义查询条件，有可能为null（用户未输入或未选择）
@Test
public void test09(){
    String username = "a";
    Integer ageBegin = null;
    Integer ageEnd = 24;
    LambdaQueryWrapper<User> queryWrapper = new LambdaQueryWrapper<>();
    queryWrapper
            .like(StringUtils.isNotBlank(username), User::getName,username)
            .ge(ageBegin != null, User::getAge,ageBegin)
            .le(ageEnd != null, User::getAge,ageEnd);
    List<User> list = userMapper.selectList(queryWrapper);
    list.forEach(System.out::println);
}

//LambdaUpdateWrapper
//将用户名中包含有a并且（年龄大于20或邮箱为null）的用户信息修改
@Test
public void test10(){
    LambdaUpdateWrapper<User> updateWrapper = new LambdaUpdateWrapper<>();
    updateWrapper
            .set(User::getName, "小黑")
            .set(User::getEmail, "xiaohei@atguigu.com")
            .like(User::getName, "a")
            .and(i -> i.gt(User::getAge, 20).or().isNull(User::getEmail));//Lambda
    int result = userMapper.update(null, updateWrapper);
    System.out.println("result="+result);
}
```





## 3、扩展功能

### 3.1. 静态工具(Db 类)

有的时候 Service 之间也会相互调用，为了避免出现循环依赖问题，MybatisPlus 提供一个静态工具类：`Db`，其中的一些静态方法与 `IService`中方法签名基本一致，也可以帮助我们实现 CRUD 功能（用不太懂）：

需求：根据 id 用户查询的接口，查询用户的同时返回用户收货地址列表

```java
// service 层的实现方法
@Override
public UserVO queryUserAndAddressById(Long userId) {
    // 1.查询用户
    User user = getById(userId);
    if (user == null) {
        return null;
    }
    // 2.查询收货地址
    List<Address> addresses = Db.lambdaQuery(Address.class)
            .eq(Address::getUserId, userId)
            .list();
    // 3.处理vo
    UserVO userVO = BeanUtil.copyProperties(user, UserVO.class);
    userVO.setAddresses(BeanUtil.copyToList(addresses, AddressVO.class));
    return userVO;
}
```

在查询地址时，采用了 Db 的静态方法，因此避免了注入 AddressService，减少了循环依赖的风险。

使用 Db 类的注意事项：只有在版本较新的 MybatisPlus 中才支持该类，我使用的是 `3.5.3.1` 。

### 3.2. 通用枚举

比如 User 类中有一个用户状态字段：

`private Integer status; 表示使用状态，1 正常 2 冻结`

像这种字段一般会定义一个枚举，做业务判断的时候就可以直接基于枚举做比较。但是数据库采用的是 `int`类型，对应的 PO 也是 `Integer`。因此业务操作时必须手动把 `枚举`与 `Integer`转换，非常麻烦。

因此，MybatisPlus 提供了一个处理枚举的类型转换器，可以 **把枚举类型与数据库类型自动转换** 。

#### 3.2.1. 定义枚举

1. 首先定义一个用户状态的枚举类：

```java

package com.xxx.xx.enums;

import com.baomidou.mybatisplus.annotation.EnumValue;
import lombok.Getter;

@Getter
public enum UserStatus {
    NORMAL(1, "正常"),
    FREEZE(2, "冻结")
    ;
    @EnumValue
    private final int value;
    @JsonValue
    private final String desc;

    UserStatus(int value, String desc) {
        this.value = value;
        this.desc = desc;
    }
}
```

2. 然后把 `User`类中的 `status`字段改为 `UserStatus` 类型：

`private UserStatus status; 表示使用状态，1 正常 2 冻结`

3. 要让 `MybatisPlus`处理枚举与数据库类型自动转换，必须告诉 `MybatisPlus`，枚举中的哪个字段的值作为数据库值。
   `MybatisPlus`提供了 `@EnumValue`注解来标记枚举属性，因此要在 value 属性上添加 ` @EnumValue` 注解。

#### 3.2.2. 配置枚举处理器

在 application.yaml 文件中添加配置：

```yaml
mybatis-plus:
  configuration:
    default-enum-type-handler: com.baomidou.mybatisplus.core.handlers.MybatisEnumTypeHandler
```

#### 3.2.3. 测试

```java
@Test
void testService() {
    List<User> list = userService.list();
    list.forEach(System.out::println);
}
```

返回的结果中，status 字段值为：`status="NORMAL" `，但是将该值返回给前端并不见名知意，因此为了保证格式的统一，需要在 UserStatus 枚举中通过 `@JsonValue`注解标记 JSON 序列化时展示的字段：

此时再次测试会发现 `status="正常"`达到了预期效果。

## 4、插件功能

### 4.1. 分页插件

在未引入分页插件的情况下，`MybatisPlus`是不支持分页功能的，`IService`和 `BaseMapper`中的分页方法都无法正常起效。所以，必须要配置分页插件。

#### 4.1.1. 配置分页插件

在项目中新建配置类（MybatisConfig）：

```java
package com.xxx.xx.config;

import com.baomidou.mybatisplus.annotation.DbType;
import com.baomidou.mybatisplus.extension.plugins.MybatisPlusInterceptor;
import com.baomidou.mybatisplus.extension.plugins.inner.PaginationInnerInterceptor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MybatisConfig {

    @Bean
    public MybatisPlusInterceptor mybatisPlusInterceptor() {
        // 初始化核心插件
        MybatisPlusInterceptor interceptor = new MybatisPlusInterceptor();
        // 添加分页插件
        interceptor.addInnerInterceptor(new PaginationInnerInterceptor(DbType.MYSQL));
        return interceptor;
    }
}
```

#### 4.1.2. 分页 API

```Java
@Test
void testPageQuery() {
    // 1.分页查询，new Page()的两个参数分别是：页码、每页大小
    Page<User> p = userService.page(new Page<>(2, 2));
    // 2.总条数
    System.out.println("total = " + p.getTotal());
    // 3.总页数
    System.out.println("pages = " + p.getPages());
    // 4.数据
    List<User> records = p.getRecords();
    records.forEach(System.out::println);
}
```

这里用到了分页参数，Page，即可以支持分页参数，也可以支持排序参数。常见的 API 如下：

```java
int pageNo = 1, pageSize = 5;
// 分页参数
Page<User> page = Page.of(pageNo, pageSize);
// 排序参数, 通过OrderItem来指定
page.addOrder(new OrderItem("balance", false));

userService.page(page);
```

### 4.2 通用分页实体

#### 4.2.1. PageQuery 实体

```Java
package com.xxx.xx.domain.query;

import com.baomidou.mybatisplus.core.metadata.OrderItem;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import lombok.Data;

@Data
public class PageQuery {
    private Integer pageNo;
    private Integer pageSize;
    private String sortBy;
    private Boolean isAsc;

    public <T>  Page<T> toMpPage(OrderItem ... orders){
        // 1.分页条件
        Page<T> p = Page.of(pageNo, pageSize);
        // 2.排序条件
        // 2.1.先看前端有没有传排序字段
        if (sortBy != null) {
            p.addOrder(new OrderItem(sortBy, isAsc));
            return p;
        }
        // 2.2.再看有没有手动指定排序字段
        if(orders != null){
            p.addOrder(orders);
        }
        return p;
    }

    public <T> Page<T> toMpPage(String defaultSortBy, boolean isAsc){
        return this.toMpPage(new OrderItem(defaultSortBy, isAsc));
    }

    public <T> Page<T> toMpPageDefaultSortByCreateTimeDesc() {
        return toMpPage("create_time", false);
    }

    public <T> Page<T> toMpPageDefaultSortByUpdateTimeDesc() {
        return toMpPage("update_time", false);
    }
}
```

#### 4.2.2. PageDTO 实体

```java
package com.xxx.xx.domain.dto;

import cn.hutool.core.bean.BeanUtil;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PageDTO<V> {
    private Long total;
    private Long pages;
    private List<V> list;

    /**
     * 返回空分页结果
     * @param p MybatisPlus的分页结果
     * @param <V> 目标VO类型
     * @param <P> 原始PO类型
     * @return VO的分页对象
     */
    public static <V, P> PageDTO<V> empty(Page<P> p){
        return new PageDTO<>(p.getTotal(), p.getPages(), Collections.emptyList());
    }

    /**
     * 将MybatisPlus分页结果转为 VO分页结果
     * @param p MybatisPlus的分页结果
     * @param voClass 目标VO类型的字节码
     * @param <V> 目标VO类型
     * @param <P> 原始PO类型
     * @return VO的分页对象
     */
    public static <V, P> PageDTO<V> of(Page<P> p, Class<V> voClass) {
        // 1.非空校验
        List<P> records = p.getRecords();
        if (records == null || records.size() <= 0) {
            // 无数据，返回空结果
            return empty(p);
        }
        // 2.数据转换
        List<V> vos = BeanUtil.copyToList(records, voClass);
        // 3.封装返回
        return new PageDTO<>(p.getTotal(), p.getPages(), vos);
    }

    /**
     * 将MybatisPlus分页结果转为 VO分页结果，允许用户自定义PO到VO的转换方式
     * @param p MybatisPlus的分页结果
     * @param convertor PO到VO的转换函数
     * @param <V> 目标VO类型
     * @param <P> 原始PO类型
     * @return VO的分页对象
     */
    public static <V, P> PageDTO<V> of(Page<P> p, Function<P, V> convertor) {
        // 1.非空校验
        List<P> records = p.getRecords();
        if (records == null || records.size() <= 0) {
            // 无数据，返回空结果
            return empty(p);
        }
        // 2.数据转换
        List<V> vos = records.stream().map(convertor).collect(Collectors.toList());
        // 3.封装返回
        return new PageDTO<>(p.getTotal(), p.getPages(), vos);
    }
}
```

最终，业务层的代码可以简化为：

```java
@Override
public PageDTO<UserVO> queryUserByPage(PageQuery query) {
    // 1.构建条件
    Page<User> page = query.toMpPageDefaultSortByCreateTimeDesc();
    // 2.查询
    page(page);
    // 3.封装返回
    return PageDTO.of(page, UserVO.class);
}
```

如果是希望自定义 PO 到 VO 的转换过程，可以这样做：

```
@Override
public PageDTO<UserVO> queryUserByPage(PageQuery query) {
    // 1.构建条件
    Page<User> page = query.toMpPageDefaultSortByCreateTimeDesc();
    // 2.查询
    page(page);
    // 3.封装返回
    return PageDTO.of(page, user -> {
        // 拷贝属性到VO
        UserVO vo = BeanUtil.copyProperties(user, UserVO.class);
        // 用户名脱敏
        String username = vo.getUsername();
        vo.setUsername(username.substring(0, username.length() - 2) + "**");
        return vo;
    });
}
```

# 七、Redis

Spring 通过模板方式提供了对 Redis 的数据查询和操作功能。

#### RedisTemplate 就是在一个方法中定义了一个算法的骨架，但是把进一步的步骤延迟到子类去实现，模板方法使得子类可以在不改变算法结构的情况下，重新定义算法的某些步骤。

RedisTemplate 对 Redis 中的物种基础类型，分别提供了五个子类进行操作。

```java
ValueOperations valueOperations = redisTemplate.opsForValue();
HashOperations valueOperations = redisTemplate.opsForHash();
ListOperations valueOperations = redisTemplate.opsForList();
SetOperations valueOperations = redisTemplate.opsForSet();
ZsetOperations valueOperations = redisTemplate.opsForZset();

```

# 八、Lombok

| 注解                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| @Slf4j                   | 自动生成该类的 log 静态常量                                  |
| @Log4j2                  | 注解在类上。为类提供一个 属性名为 log 的 log4j 日志对象，和@Log4j 注解类似。 |
| @Setter                  | 注解在属性上，为属性提供 setter 方法。注解在类上，为所有属性添加 setter 方法 |
| @Getter                  | 注解在属性上，为属性提供 getter 方法。注解在类上，为所有属性添加 getter 方法 |
| @EqualsAndHashCode       |                                                              |
| @RequiredArgsConstructor |                                                              |
| @NoArgsConstructor       |                                                              |
| @AllArgsConstructor      |                                                              |
| @NotNull                 |                                                              |
| @NullAble                |                                                              |
| @ToString                |                                                              |
| @Value                   | 所有变量为 final，等同于添加@Getter @ToString @EqualsAndHashCode @RequiredArgsConstructor |
| @Data                    | 等同于添加@Getter/@Setter @ToString @EqualsAndHashCode @RequiredArgsConstructor |
| @Builder                 | 自动生成流式 set 值写法                                      |

注：@EqualsAndHashCode 默认情况下，会使用所有非瞬态(non-transient)和非静态(non-static)字段来生成 equals 和 hascode 方法，也可以指定具体使用哪些属性。如果某些变量不想加入判断通过 exclude 排除，或者使用 of 指定使用某些字段

# 九、Swagger

## 1、SpringBoot 集成 Swagger

1. 添加对应的依赖
2. 新建一个配置类，添加@EnableSwagger2 和@Configuration 注解，打开并自定义配置 Swagger
3. 通过 `http://项目IP:端口/swagger-ui.html`访问 API 接口文档

[附：SpringBoot 集成 Swagger 详细教程](http://www.imooc.com/wiki/swaggerlesson/springbootswagger.html)

## 2、常用注解

Swagger 是为了解决企业中接口（api）中定义统一标准规范的文档生成工具。可以通过在代码中添加 Swagger 的注解来生成统一的 API 接口文档。注解主要有以下几种：

| 注解名称           | 使用地方         | 说明                            |
| ------------------ | ---------------- | ------------------------------- |
| @Api               | 类               | 描述后端 API 接口类级别上的说明 |
| @ApiOperation      | 方法             | 描述后端 API 接口的信息         |
| @ApiParam          | 方法、参数、字段 | 对方法、参数添加元数据          |
| @ApiModel          | 类               | 对类进行说明                    |
| @ApiModelPropery   | 方法、字段       | 对类的属性说明                  |
| @ApiIgnore         | 类、方法、参数   | Swagger 将会忽略这些            |
| @ApiImplicitParam  | 方法             | 单独请求的参数                  |
| @ApiImplicitParams | 方法             |                                 |

| 注解参数    | 类型     | 默认值 | 涉及注解      | 说明                 |
| ----------- | -------- | ------ | ------------- | -------------------- |
| value       | String   |        |               | 描述接口用途         |
| tags        | String[] |        |               | 接口分组             |
| notes       | String   |        | @ApiOpreation | 对接口做出进一步描述 |
| httpMethod  | String   |        | @ApiOpreation | 接口请求方法         |
| nickname    | String   |        | @ApiOpreation | 接口别名             |
| protocols   | String   |        |               | 接口使用的网络协议   |
| hidden      | Boolean  |        |               | 是否隐藏该接口       |
| code        | int      |        | @ApiOpreation | 接口返回状态码       |
| description |          |        | @Api          |                      |
| produces    |          |        | @Api          |                      |
| consumes    |          |        | @Api          |                      |

## 3、Swagger 配置

创建 Swagger 的配置代码如下：

```java
@EnableSwagger2
@Configuration
public Class Swagger2Config{
    @Bean
    public Docket createApiDoc(){
        return new Docket(DocumentationType.SWAGGER_2)
            .apiInfo(apiInfo())
            .select()
            .apis(RequestHandlerSelector.basePackage("your_package_name"))
            .paths(PathSelectors.any())
            .build();
    }
    private ApiInfo apiInfo(){
        return new ApiInfoBuilder()
            .title()
            .description()
            .version()
            .build();
    }
}
```

| 方法名      | 描述              |
| ----------- | ----------------- |
| title       | 填写 API 文档标题 |
| description | 填写 API 文档描述 |
| version     | 填写 API 文档版本 |
| bulid       | 创建 ApiInfo 实例 |

# 附录

## 1、Spring 注解详解

### 1）配置类注解

| 注解                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| @SpringBootApplication   | 声明让 Spring Boot 自动给程序进行必要的配置，等同于@Configuration ，@EnableAutoConfiguration 和 @ComponentScan 三个配置。 |
| @Configuration           | 说明这是一个配置类                                           |
| @EnableAutoConfiguration | Spring Boot 自动配置，尝试根据你添加的 jar 依赖自动配置你的 Spring 应用。 |
| @ComponentScan           | 自动搜索当前类所在的包以及子包，把所有标注为需要装配的的 Bean 自动创建出来。默认会装配标识了@Controller，@Service，@Repository，@Component 注解的类到 spring 容器中。如果通过注解实现装配组件，这个配置类需要位于项目的根目录，让 Spring 明白在哪里扫描。以便扫描到整个项目的组件类 |
| @Import                  | 引入带有@Configuration 的 java 类。                          |
| @ImportResourse          | 引入 spring 配置文件 applicationContext.xml                  |

注：@Configuration 注解的配置类有如下要求：

1. @Configuration 不可以是 final 类型；
2. @Configuration 不可以是匿名类；
3. 嵌套的 configuration 必须是静态类。

### 2）组件注解

| 注解           | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| @Component     | 说明这是一个交给 Spring 保管的 JAVA Bean。泛指各种组件。     |
| @Bean          | 产生一个 Bean 对象，将它交给 spring 管理，产生方法只会调用一次。 |
| @Scope         | 声明一个原型（Prototype）的 Bean 时，需要添加一个额外的 `@Scope`注解 |
| @Order         | 指明注入的 Bean 的顺序                                       |
| @PostConstruct | 定义组件初始化时运行的方法                                   |
| @PreDestroy    | 定义组件销毁前运行的方法                                     |

注：@Bean 属性说明

属性有：value、name、autowire、initMethod、destroyMethod。

- name 和 value 两个属性是相同的含义的， 在代码中定义了别名。为 bean 起一个名字，如果默认没有写该属性，那么就使用方法的名称为该 bean 的名称。
- autowire 指定 bean 的装配方式， 根据名称 和 根*[欢迎转载听雨的人博客]*据类型 装配， 一般不设置，采用默认即可。autowire 指定的装配方式 有三种 Autowire.NO (默认设置)、Autowire.BY_NAME、Autowire.BY_TYPE。
- initMethod 和 destroyMethod 指定 bean 的初始化方法和销毁方法， 直接指定方法名称即可，不用带括号。

### 3）注入注解

| 注解       | 说明                                                   |
| ---------- | ------------------------------------------------------ |
| @Resource  | 按名称注入                                             |
| @Autowired | 按类型注入                                             |
| @Inject    | 按类型注入                                             |
| @Value     | 将常量、配置文件中的值、其他 bean 的属性值注入到变量中 |

#### @Resource

需要 JAVA Bean 注入时使用，可以写到字段和 setter 方法上，选其一即可。Resource 默认按照名称自动注入，属性 name 解析为 bean 的名字，type 解析为 bean 的类型。注入规则如下：

- 如果同时指定了 name 和 type，则从 Spring 上下文中找到唯一匹配的 bean 进行装配，找不到则抛出异常。
- 如果指定了 name，则从上下文中查找名称（id）匹配的 bean 进行装配，找不到则抛出异常。
- 如果指定了 type，则从上下文中找到类似匹配的唯一 bean 进行装配，找不到或是找到多个，都会抛出异常。
- 如果既没有指定 name，又没有指定 type，则自动按照 byName 方式进行装配；如果没有匹配，则回退为一个原始类型进行匹配，如果匹配则自动装配。

#### @Autowired

需要 JAVA Bean 注入时使用，可以写到字段和 setter 方法上，选其一即可。但是 `Autowired`只按类型注入，默认情况下要求依赖的对象必须存在，如果允许 null 值，需要设置属性 required=false，如果需要按名称来装配，需要和@Qualifier 注解一起使用。

#### @Inject

需要 JAVA Bean 注入时使用，可以作用在变量、setter 方法、构造函数上。默认根据类型 type 进行自动装配的，如果需要按名称进行装配，则需要配合@Named。

#### @Value

通过注解将常量、配置文件中的值、其他 bean 的属性值注入到变量中，作为变量的初始值。bean 属性、系统属性、表达式注入，使用@Value("#{}")。bean 属性注入*[Power By 听雨的人]*需要注入者和被注入者属于同一个 IOC 容器，或者父子 IOC 容器关系，在同一个作用域内。配置文件属性注入@Value*[Power By 听雨的人]*("${}")

### 4）MVC 注解

| 注解               | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
| @Controller        | 负责处理由 DispatcherServlet 分发的请求，把用户请求的数据经过处理封装成一个模型，然后再把这个模型返回给对应的视图进行展示。Controller 不会直接依赖于 HttpServletRequest 和 HttpServletResponse 等 HttpServlet 对象，它们可以通过 Controller 的方法参数灵活的获取到。 |
| @Service           | 修饰 MVC 中 Service 层的组件                                 |
| @Repository        | 注解 DAO 层（Mapper 层）                                     |
| @RequestBody       | 修饰返回的数据，当返回的数据不是 html 标签的页面，而是其他某种格式的数据时（如 json、xml 等）使用。 |
| @RestController    | 相当于@Controller 和@ResponseBody                            |
| @RequestMapping    | 是用来处理请求地址映射的注解，可以用于类或者方法上。用在类上表示类中所有响应请求的方法都是以该地址作为父路径。一共有六个属性。 |
| @RequestParam      | 获取前端请求传来的参数，有三个属性：defaultValue 表示设置默认值，required 通过 boolean 设置是否是必须要传入的参数，value 值表示接受的传入的参数类型。 |
|                    |                                                              |
| @ModelAttribute    |                                                              |
| @SessionAttributes |                                                              |
| @PathVarible       |                                                              |

注：@RequestMapping 的六个属性

1. value：指定请求的实际地址（默认属性）
2. method：指定请求的类型，GET POST DELETE PUT
3. consumes：指定处理请求的 Content-Type(内容类型)
4. produces：指定返回的内容类型，仅当请求包含该类型时才回返回相应的数据
5. params：指定请求必须包含某些参数值才会处理该请求
6. headers：指定请求必须包含某些指定的 header 值才会处理该请求

### 5）AOP 切面注解

Spring 支持 AspectJ 的注解式 aop 编程，需要在 java 的配置类中使用@EnableAspectJAutoProxy 注解开启 Spring 对 AspectJ 代理的支持。

| 注解                   | 说明                                                         |
| ---------------------- | ------------------------------------------------------------ |
| @EnableAspectAutoProxy | 开启 Aspect 代理，使用 AOP 注解必备                          |
| @Aspect                | 声明一个切面类，该类中的方法都会在合适的时机中插入到需要该方法的地方，方法也需要注解标识 |
| @Before                | 在指定方法执行前执行此方法，需要在注解参数中传入指定方法全名 |
| @After                 | 在指定方法执行后执行此方法                                   |
| @AfterRunning          | 在方法返回结果后执行此方法                                   |
| @AfterThrowing         | 在方法抛出异常后执行此方法                                   |
| @Around                | 围绕着方法执行                                               |
| @PointCut              |                                                              |
