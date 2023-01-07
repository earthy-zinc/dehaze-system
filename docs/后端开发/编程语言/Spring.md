# 一、Spring FrameWork

## 1、`IoC`容器

### 1）容器的概念

容器是为某些特定组件对象提供必要支持的一个软件环境。他提供了一些底层服务，让容器所承载的对象不必在考虑这些问题。Tomcat就是一个Servlet容器，底层实现了TCP连接，解析HTTP协议等非常复杂的服务。我们自己就无需在组件中编写这些复杂的逻辑。IoC容器它可以管理所有轻量级的JavaBean组件，提供的底层服务包括组件的生命周期管理、配置和组装服务、AOP支持，以及建立在AOP基础上的声明式事务服务等。

### 2）控制反转IoC

`IoC`意为控制反转（Inversion of Control），对程序中对象的创建、配置这样的控制权由应用程序转移到了`IoC`容器，那对于某个具体的实例对象它的所有组件对象不再由应用程序自己创建和配置，而是通过`IoC`容器负责。这样应用程序能够直接使用已经创建并配置好的组件。

在设计上`IoC`容器是一个无侵入的容器，应用程序的组件无需实现Spring的特定接口，那么这些组件即可以在spring容器中运行，又能够自己编写代码组装他所需要的对象。还有就是在测试的时候，也就不需要实现接口，不依赖与Spring容器，可单独测试。

### 3）依赖注入

这些组件需要通过注入机制来装入到实例对象中，供实例对象使用。依赖注入的方式可以有两种，一种是通过`setXXX()`方法注入，另一种是通过构造方法实现。Spring的IoC容器同时支持属性注入和构造方法注入，并允许混合使用。

因为`IoC`容器需要负责实例化所有组件对象，所以需要告诉容器如何创建组件对象，以及各个组件对象之间的依赖关系，即装配方式。在Spring可以通过两种方式实现，一种是XML配置文件，另一种是通过注解。

### 4）组件装配

#### I   通过XML装配组件

我们需要自己将组件之间的依赖关系描述出来，然后交给容器来创建并装配。

**第一步 编写配置文件application.xml**

我们需要编写一个特定的名叫application的配置文件`application.xml`告诉Spring容器应该如何创建、并按顺序正确的注入到相应的组件中。Bean表示这是一个Java Bean或者说是一个组件。id唯一标识了一个Java Bean，class提供了文件路径。每个Java Bean内部可以有一个或多个需要注入的属性，以property标签表示。而这些属性也是一个Java Bean，name表示在这个组件内部这个需要注入的属性的名称是什么。ref表示这个需要注入的属性所指向的Java Bean的id。这些Java Bean在配置文件的相对位置并不重要，但是每个组件中要注入的属性需要写全，不写全的画spring会漏掉注入该属性。如果注入的不是Java Bean那么将ref改为value。

总结来说，Java Bean通过引用注入，数据类型通过value注入。

```xml
<bean id="userService" class="com.itranswarp.learnjava.service.UserService">
    <property name="mailService" ref="mailService" />			<!--引用注入-->
    <property name="username" value="root" />				    <!--值注入-->
    <property name="password" value="password" />
</bean>
```

**第二步 在代码中加载配置文件**

我们需要创建一个`Spring IoC`容器的实例，然后加载配置文件。接下来我们就可以从Spring容器中取出组件并使用它。Spring容器命名为应用程序上下文，就是`ApplicationContext`，它是一个接口，用来加载配置文件，有很多实现类。通过xml加载需要`ClassPathXmlApplicationContext`实现类来帮我们自动从项目路径下查找指定的配置文件，参数为配置文件名。通过注解加载需要`AnnotationConfigApplicationContext`实现类，参数为配置类名称，必须传入一个标注了`@Configuration`的类名。。

#### II   通过注解装配组件

见组件详解

## 2、AOP

在实际开发中有很多功能是许多组件通用的，但又是非核心的业务逻辑。让框架把一些很多个不同的组件之间通用的非核心的业务逻辑通过某种方法，织入到组件中。那么AOP要把切面即一些非核心、但又必要的逻辑织入核心逻辑中，我们在调用某个业务方法时，spring会对该方法进行拦截，并在拦截前后进行安全检查、日志、事务等处理。从而完成了整个业务流程。有3种方式实现。

* 编译期，由编译器把切面（非核心的逻辑）编译进字节码。
* 类加载器：当目标被装载到JVM时，通过一个特殊的类加载器，对目标类的字节码重新增强
* 运行期：通过动态代理实现运行期动态织入。

Spring的AOP实现就是基于JVM的动态代理，通过AOP技术，可以让我们把一些常用的功能如权限检查、日志、事务等，从每个业务方法中剥离出来。

我们使用AOP非常简单，一共需要三步：

1. 定义切入方法，并在方法上通过AspectJ的注解告诉Spring应该在何处调用此方法；
2. 在需要切入方法的地方标记`@Component`和`@Aspect`；
3. 在`@Configuration`类上标注`@EnableAspectJAutoProxy`。

我们还可以通过自定义注解来切入功能。在那些需要切入这种常用的功能的方法头上，标记一个自定义注解，而在切入方法（常用的功能逻辑所在的方法）的AOP注解参数中填入该注解的名称，参数格式为`"@annotation(your_annotation_name)"`，那么只要标注了你自定义注解的地方，spring都会把切入方法切入到里面。

# 二、Spring Web

## 1、Controller层

在MVC模式中，controller作为控制器，控制视图和模型之间的交流，使视图和模型分离开。在Web应用中，也是类似的概念，控制层接受前端发来的请求，交由对应的服务层处理，然后返回响应结果。在整个架构中，我们大致上可以把Controller看做是前端和后端交互的中介，由于前端发来的请求多种多样，后端负责处理这些请求的类以及方法也都不同，因此我们需要一个中间商，接收前端发来的请求，先对其进行简单的处理，识别该请求的意图，然后交由对应的方法去处理。Controller层因此有着承前启后的作用。

HTTP请求分为请求行、请求头、请求体三部分。请求行中携带了请求方法、URL、HTTP协议版本。请求头中携带了HTTP请求的一些必要信息，而请求体中是请求所携带的数据。每一个请求向服务器请求的数据都不太一样，因此请求行、请求头、请求体中的内容也不太一样。服务器要根据这些请求返回不同的数据，首先就是要分辨这些请求到底是想请求什么。

Web应用中Controller负责接收HTTP请求，那么Controller层就需要对请求进行分析处理。分析HTTP请求的意图，然后交由Service层去处理。在Controller层中，我们有两大任务，获取请求信息、返回响应数据。为了处理好这两大任务，我们划分出以下几个步骤：

1. 首先，请求是多种多样的，单一的Controller无法满足所有请求的要求。我们先把请求分类，不同的URL对应着不同种类的请求。URL是有层级的，我们可以对请求的种类再进一步细分。因此我们设置了不同的Controller类、不同的Controller方法、来处理不同种类的请求，这时候就需要**指定请求的映射规则**。
2. 其次，对于同一类的请求，我们就具体的了解请求的数据是什么，那么我们就需要**获取请求参数**或者**获取请求体**，来进一步识别请求是想要获取哪些数据。
3. 还有一点，对于服务器中某些私密的资源，我们不可能让任意的请求都能获取到，因此需要识别HTTP请求的身份，是否具有相关的权限去获取资源。那么我们就需要**获取请求头**。
4. 如果上面的方法不能够满足我们需要的话，我们可以直接获取封装在一个对象中的HTTP请求的全部信息，这叫做**获取原生对象**。
5. 

### 1）指定请求映射规则

@RequestMapping用于映射前端HTTP发来的Request请求，对于前端发来的不同请求，我们应该指定不同的Controller、不同的方法来处理。RequestMapping注解就是让我们设置HTTP请求映射到对应controller方法上的相关规则，比如说指定一个controller方法处理的请求路径、请求方式、请求参数等等一系列配置。施加@RequestMapping注解上配置的参数会限制HTTP请求映射到该方法上的范围。

@RequestMapping注解参数说明

| 参数         | 值       | 说明                                                         |
| ------------ | -------- | ------------------------------------------------------------ |
| name         | String   | 为该RequestMapping设置一个名字                               |
| value / path | String[] | 指定接收的URI路径。支持Ant样式路径匹配方法，yml占位符如.`${path}` |
| method       | emum[]   | 指定接收的请求方法。`public enum RequestMethod {GET,HEAD,POST,PATCH,DELETE,OPTIONS,TRACE}` |
| params       | String[] | 指定接受的请求参数。只有HTTP请求带有对应的参数时，才会被该Controller处理，使用`!`表示不能具有该类请求。 |
| header       | String[] | 指定接收的请求头。具有某些请求头或者某些请求头有特定的值，才会被该Controller处理，使用`!`表示不能具有该类请求头。 |
| consumes     | String[] | 指定接收的请求内容类型Content-Type                           |
| produces     | String[] | 指定从HTTP请求中发来的可接受响应的Content-Type               |

注：

1、注意到@RequestMapping可以使用在类上和方法上，在方法上的@RequestMapping会继承类上已有的设置。

2、Ant样式路径匹配方法

| 路径 | 说明                        | 实例                                                         |
| ---- | --------------------------- | ------------------------------------------------------------ |
| ?    | 匹配任意单个字符，不包含`/` | `/p?ttern`匹配该文件夹下符合该规则的的文件夹（不包含子文件夹） |
| *    | 匹配0或者任意数量的字符     | `/*.jsp`匹配当前文件夹下任何JSP文件（不包含子文件夹）        |
| **   | 匹配0或者更多的目录         | `/**/*.jsp`匹配该文件夹及其子文件夹任何JSP文件               |

### 2）获取请求参数

#### 获取路径参数

@PathVariable用来获取通过URL路径传递的请求参数，通常添加在Controller方法的参数中，Controller方法所映射的路径中需要写明通过路径传递了哪些参数。@PathVariable注解参数有两个，分别是value：映射请求路径参数，required：请求路径参数是否必须

```java
@RequestMapping("/user/{id}/{name}")
public String findUser(@PathVariable("id")  Integer id,
                       @PathVariable("name") String name){
    // TODO
}
```

#### 获取请求体中JSON格式参数

@RequestsBody用来获取请求体中的JSON数据，并将JSON数据转化为JAVA对象，需要JSON数据属性名和JAVA对象变量名一一对应，才回将数据传递到Java对象中，否则无法获取对应的请求数据。

注意：使用@RequestsBody获取请求体数据，需要请求头中的 Content-Type 值为application/json否则会无法获取。

#### 获取QueryParameter格式参数

@RequestParam用于获取QueryParameter格式参数。类似于`URI?name1=value1&name2=value2`格式在URL上传输的参数叫做QueryParameter格式参数，默认情况下，Controller映射到的请求参数都是QueryParameter类型的参数，且需要请求中参数名和Controller方法中变量名一一对应，才能映射成功。

**总结：**通过`@RequestsBody`和`@RequestParam`两个注解，我们可以直接单独获取每一个请求参数，也可以将参数封装到自定义实体对象中，实体类中的成员变量要和请求参数名对应上。并且要提供对应的set/get方法。

#### `@RequestsBody`和`@RequestParam`注解的其他属性

| 属性         | 值      | 说明                                     |
| ------------ | ------- | ---------------------------------------- |
| required     | boolean | 请求参数是否必须传入                     |
| defaultValue | String  | 如果没有传入对应请求参数，指定一个默认值 |

#### 参数类型转换

// TODO

### 3）获取请求头和cookie

@RequestsHeader用于获取请求头信息，在注解中填写请求头名称我们就可以获取到对应请求头的值

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

 @CookieValue用于获取cookie信息，使用方法和@RequestsHeader注解类似，在注解中填写cookie的名称我们就可以获取到对应cookie的值

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

我们之前使用servlet的时候，Controller获取的是request对象，response，session对象等。SpringMVC帮助我们简化了对请求信息的处理，因此我们可以通过一些注解直接获取到我们想要的信息。但是SpringMVC中也提供了获取这些原生对象的方法，只需要在方法上添加对应类型的参数就行。SpringMVC会把我们需要的对象传给我们的形参。不过这时候我们就需要使用servlet的API来处理这些数据，会稍显繁琐和麻烦。通常在我们需要设置响应头或者进行文件传输时会获取原生的对象，数据传输只需要写入响应体就可以了。

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

我们通过添加@ResponseBody注解就可以返回JSON格式的响应体，springMVC会为我们自动将Java对象转化为JSON

### 6）文件传输

#### 文件上传

HTTP请求需要满足条件：1、请求方式为POST。2、请求头Content-Type为multipart/form-data

SpringMVC接收文件：需要Controller方法中的参数为MutipartFile类型的。该类型有以下几种常见的方法

| 方法 | 说明 |
| ---- | ---- |
|      |      |

#### 文件下载

SpringMVC封装HTTP响应需要的条件：1、设置响应头的Content-Type属性为对应文件的MIME类型。2、设置响应头的Content-Disposition。3、文件数据以二进制形式写入响应体中。

## 2、拦截器



![image-20220426221837696](E:\StudyDoc\同步空间\4.阅读笔记\图片\image-20220426221837696.png)



## 3、异常处理

SpringMVC为我们提供了注解@ControllerAdvice声明一个类为spring管理的一个组件，可以为特定的Controller添加“通知”。是AOP原理的实现，也就是说将@ControllerAdvice中声明的方法织入到Controller中。

@ExceptionHandler用于捕获Controller中抛出的异常，与@ExceptionHandler注解配合，我们可以通过自定义的拦截规则在Controller发生异常之后进行拦截，在拦截之后，我们转而通过自定义的方法来继续拦截后的处理，从而返回给前端自定义的异常信息。

默认情况下，@ControllerAdvice会在发生异常后拦截所有的Controller然后进行处理。@RestControllerAdvice会将返回值写入响应体中，相当于@ControllerAdvice + @ResponseBody 。总结来说可以通过@ControllerAdvice和@ExceptionHandler实现全局的异常处理。

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

## 4、SpringMVC执行流程

1. 用户发起请求被DispatchServlet所处理
2. DispatchServlet通过HandlerMapping根据HTTP请求内容查找能够处理这个请求的Handler（Controller）。HandlerMapping就是用来处理HTTP请求和处理方法之间的映射关系
3. HandlerMapping返回一个能够处理请求的执行链给DispatchServlet，包含了Handler方法和拦截器
4. HandlerAdapter执行对应的Handler方法，把HTTP请求数据转换成合适的类型，作为Handler的方法参数传入
5. Handler方法执行完成之后的返回值被放到响应体中，然后返回给DispatchServlet，然后发送响应数据

# 三、Spring Boot



# 四、Spring Security

## 1、概述

### 认证与授权

一般的Web应用都需要对用户进行认证和授权两个操作。

认证：Authentication 验证当前访问系统的用户是不是本系统的用户，并且需要确认具体是哪一个用户。

授权： Authorization 对经过认证后的用户判断它是否有权限进行某个操作

### 一般流程

1. 当用户登录时，前端将用户输入的用户名密码传输到后台，后台使用一个类对象将其封装起来，在Spring Security中使用的是`UsernamePasswordAuthenticationToken`类
2. 程序需要负责验证前端传来的这个对象，验证方法就是调用service服务——根据用户名从数据库中取出用户信息到实体类的实例中，然后比较两者密码，如果密码正确就成功登录。登录成功的同时把包含着用户信息如用户名、密码、用户具有的权限等信息的一个对象放到类似于Session的对象中（SecurityContextHolder）
3. 用户访问一个资源的时候，首先要判断该资源是否是限制访问的资源，如果是的化，需要判断当前用户是否登录，没有登录就跳转到登录页面。
4. 如果用户已经登陆，访问某一个限制访问的资源，程序要根据用户请求的地址（url）从数据库中取出该资源对应的所有可以访问该资源的角色，与当前用户所对应的角色一一对比，判断用户是否可以访问。

![[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-LKooVd3p-1648311184478)(C:\Users\30666\AppData\Roaming\Typora\typora-user-images\image-20220326160405082.png?lastModify=1648282919)]](https://img-blog.csdnimg.cn/993321379a3843f2bb0cb8df913acb2a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5LiD5a-75YyX6YeM,size_20,color_FFFFFF,t_70,g_se,x_16)

用户提交的用户名和密码首先经过了一个认证过滤器，在这里将请求信息封装成一个类——Authentication类，这个类会存放用户信息，并且之后会存放用户的认证结果。然后调用认证方法到下一个地方进行认证。（也就是说判断是否是当前系统的用户）这里会通过一个认证管理器，认证管理器委托数据库访问认证提供者进行认证，这个提供者主要负责认证逻辑处理工作，首先从`UserDetailsService`用户详情服务商中调用`loadUserByUsername()`获取用户信息，会返回的用户详情信息类实例`UserDetails`，这个提供者通过密码加密器对比前端传来的密码和数据库中存在的密码是否一致。然后根据比对结果填充封装后的请求信息类。随后，过滤器便收到了认证的结果，过滤器将结果保存在服务器的一个叫做`SecurityContextHolder`的地方。通过这样一个方法`SecurityContextHolder.getContext().setAuthentication( authentication )`将认证结果保存。

部分方法说明：

1. `UsernamePasswordAuthenticationFilter`是处理用户名和密码认证方式的一个过滤器，将请求信息封装为Authentication类
2. `AuthenticationManager`是一个接口，定义了认证的方法，也是认证的入口，在这里他是用来管理各种认证方法的。有一个实现类ProviderManager里面会有一列表的认证方式。我们在自己写的登录认证Service中需要注入这样的接口，调用他的认证方法进行认证。会返回认证信息Authentication。我们可以对返回的认证信息进行后续的操作，如获取其中的用户信息构造jwtToken，然后将其存放在redis中
3. UserDetailsService接口：在这里我们需要指定相应的从数据库查询用户信息的方法。我们需要自己实现这个接口，注入一个DAO然后将其从数据库中查询到的用户信息转换成Security中要求的UserDetails对象即可。
4. UserDetails接口：这个接口存放了用户信息，以及与用户认证和授权有关的方法，Security会调用这个接口的实现类来判断账户的认证授权情况。

## 2、认证

## 3、授权

## 4、自定义失败处理

## 5、跨域

## 6、自定义权限校验方法

## 7、CSRF

## 8、认证处理器

# 五、Mybatis

## 1、介绍

![mybatis架构](E:\StudyDoc\同步空间\4.阅读笔记\图片\mybatis-1.png)

我们把mybatis的功能架构分为了三层：

1. 接口层：提供给程序员调用的接口API，程序员通过这些API来操作数据库，接口层收到调用请求就会转而调用数据处理层来完成具体的数据处理。
2. 数据处理层：负责对具体的从参数映射、SQL语句的解析、执行、以及执行结果的映射处理等，主要的作用就是根据调用的请求完成一次数据库的操作。
3. 基础支撑层：负责最基础的功能支撑，包括连接池管理、事务管理、配置的加载、缓存处理。这些都是进行数据库查询通用的东西，我们把他从数据库操作中抽取出来作为最基本的组件，为上层的数据处理层提供最基础的支撑。

最后是引导层，引导层不参与SQL语句的处理，它只是为了在应用程序中配置mybatis的各种功能如何去运行。

## 2、总体框架设计

### 1）接口层

接口层是mybatis提供给程序员调用的应用程序接口，我们通过接口层就可以方便的调用数据库获取数据。目前mybatis提供两种方式与数据库进行交互。

* 创建SqlSession对象。第一种是创建一个SQL语句会话对象（SqlSession）使用该对象完成对数据库的交互，该对象内部维护了与数据库的连接，提供了与数据库进行增删改查操作的方法。
* 使用Mapper接口。第二种是通过接口调用的方式，mybatis中应用程序的某个对象与数据库某个表连接的桥梁是通过mapper映射实现的，配置文件中的每一个mapper结点都对应着一个mapper接口。接口中的每一个方法对应这配置文件中每一条SQL语句。我们在配置好以后，mybatis会根据相应的接口方法通过动态代理生成一个mapper实例，我们在调用mapper接口的某一个方法的时候，mybatis会根据这个方法的名称还有参数类型，确定SQL语句，实现对数据库的操作。

### 2）数据处理层

我们在Java中调用这些对数据库增删改查方法的时候，会传入一些参数，这些参数可能是具体要存的数据，或者是其他的东西，mybatis的数据处理层所要实现的功能就是从这里展开的，主要完成两个功能

* 一是通过传入的参数构建动态的SQL语句
* 二是执行对应的SQL语句并封装查询结果映射到Java对象中。

### 3）框架支撑层

框架支撑层，负责数据库查询中一些通用的东西，主要有以下几点内容

* 事务管理机制：
* 连接池管理机制：
* 缓存机制：为了减少数据库的压力，提高数据利用率，mybatis会对一些查询结果缓存到本地中，在一定的时间间隔内多次的同一查询，mybatis会直接返回缓存的内容，而不会再去数据库中查找。
* SQL语句的配置方式：Java程序中SQL语句配置方式有两种，一种是通过XML来配置，另一种是在mapper接口上使用注解来配置。这个功能就是为了识别并处理两种不同的配置方式而存在的。

## 3、实现功能的层次结构

1. 我们使用mybatis查询数据库，首先就是要创建一个SQL会话对象，也就是SqlSession，创建完成之后，就开启了一个与数据库的连接会话，我们可以通过这个对象，来执行SQL语句、提交或者回滚事务。
2. 但是实际上出于分离职责、防止一个对象身兼太多职责，SqlSession只是执行数据库查询的第一层对象，它会紧接着调用Executor对象，这个对象会负责SQL动态语句的生成，对查询出来的结果进行缓存，对这些结果进行维护，定期删除等。
3. 其次是语句处理器对象，由于Java程序在底层与数据库的交互是通过JDBC实现的，mybatis是在JDBC的基础上做出了进一步的封装。因此语句处理器对象StatementHandler主要负责与JDBC语句之间的交互。设置语句参数，将返回的结果映射到Java对象。
4. 接下来就是JDBC层，是真正负责查询数据库的东西。

![实现功能而层次结构](E:\StudyDoc\同步空间\4.阅读笔记\图片\mybatis-2.png)

### 4、mybatis初始化

我们如果想要在自己的程序中引入一个插件或者是框架，单单只把软件包导入进来是没有作用的，我们要在程序中使用它就需要进行一系列的配置，就比如Java的那些内置工具，他就在那但我们不能直接使用，我们调用的时候新建该对象，需要传入一些参数。类比到框架的初始化及配置上，就是这个道理。

mybatis初始化的方式主要有两种：一种是通过XML配置。第二种是基于Java的API

### 1）XML配置初始化

### 2）Java的API初始化










# 六、Mybatis Plus

## 1、在项目中引入Mybatis Plus

* 第一步，添加相应依赖
* 第二步，需要在 Spring Boot 启动类中添加 `@MapperScan` 注解，扫描 Mapper 文件夹
* 第三步，编写 Mapper 包下的接口，继承Mybatis Plus提供的`BaseMapper<T>`

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

# 七、Redis

Spring通过模板方式提供了对Redis的数据查询和操作功能。

RedisTemplate就是在一个方法中定义了一个算法的骨架，但是把进一步的步骤延迟到子类去实现，模板方法使得子类可以在不改变算法结构的情况下，重新定义算法的某些步骤。

RedisTemplate对Redis中的物种基础类型，分别提供了五个子类进行操作。

```java
ValueOperations valueOperations = redisTemplate.opsForValue();
HashOperations valueOperations = redisTemplate.opsForHash();
ListOperations valueOperations = redisTemplate.opsForList();
SetOperations valueOperations = redisTemplate.opsForSet();
ZsetOperations valueOperations = redisTemplate.opsForZset();

```




#  八、Lombok

| 注解                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| @Slf4j                   | 自动生成该类的log静态常量                                    |
| @Log4j2                  | 注解在类上。为类提供一个 属性名为log 的 log4j 日志对象，和@Log4j注解类似。 |
| @Setter                  | 注解在属性上，为属性提供setter方法。注解在类上，为所有属性添加setter方法 |
| @Getter                  | 注解在属性上，为属性提供getter方法。注解在类上，为所有属性添加getter方法 |
| @EqualsAndHashCode       |                                                              |
| @RequiredArgsConstructor |                                                              |
| @NoArgsConstructor       |                                                              |
| @AllArgsConstructor      |                                                              |
| @NotNull                 |                                                              |
| @NullAble                |                                                              |
| @ToString                |                                                              |
| @Value                   | 所有变量为final，等同于添加@Getter @ToString @EqualsAndHashCode @RequiredArgsConstructor |
| @Data                    | 等同于添加@Getter/@Setter @ToString @EqualsAndHashCode @RequiredArgsConstructor |
| @Builder                 | 自动生成流式 set 值写法                                      |

注：@EqualsAndHashCode默认情况下，会使用所有非瞬态(non-transient)和非静态(non-static)字段来生成equals和hascode方法，也可以指定具体使用哪些属性。如果某些变量不想加入判断通过exclude排除，或者使用of指定使用某些字段

# 九、Swagger

## 1、SpringBoot集成Swagger

1. 添加对应的依赖
2. 新建一个配置类，添加@EnableSwagger2和@Configuration注解，打开并自定义配置Swagger
3. 通过`http://项目IP:端口/swagger-ui.html`访问API接口文档

[附：SpringBoot集成Swagger详细教程](http://www.imooc.com/wiki/swaggerlesson/springbootswagger.html)

## 2、常用注解

Swagger是为了解决企业中接口（api）中定义统一标准规范的文档生成工具。可以通过在代码中添加Swagger的注解来生成统一的API接口文档。注解主要有以下几种：

| 注解名称           | 使用地方         | 说明                          |
| ------------------ | ---------------- | ----------------------------- |
| @Api               | 类               | 描述后端API接口类级别上的说明 |
| @ApiOperation      | 方法             | 描述后端API接口的信息         |
| @ApiParam          | 方法、参数、字段 | 对方法、参数添加元数据        |
| @ApiModel          | 类               | 对类进行说明                  |
| @ApiModelPropery   | 方法、字段       | 对类的属性说明                |
| @ApiIgnore         | 类、方法、参数   | Swagger将会忽略这些           |
| @ApiImplicitParam  | 方法             | 单独请求的参数                |
| @ApiImplicitParams | 方法             |                               |

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

## 3、Swagger配置

创建Swagger的配置代码如下：

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

| 方法名      | 描述            |
| ----------- | --------------- |
| title       | 填写API文档标题 |
| description | 填写API文档描述 |
| version     | 填写API文档版本 |
| bulid       | 创建ApiInfo实例 |

# 附录

## 1、Spring注解详解

### 1）配置类注解

| 注解                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| @SpringBootApplication   | 声明让Spring Boot自动给程序进行必要的配置，等同于@Configuration ，@EnableAutoConfiguration 和 @ComponentScan 三个配置。 |
| @Configuration           | 说明这是一个配置类                                           |
| @EnableAutoConfiguration | Spring Boot自动配置，尝试根据你添加的jar依赖自动配置你的Spring应用。 |
| @ComponentScan           | 自动搜索当前类所在的包以及子包，把所有标注为需要装配的的Bean自动创建出来。默认会装配标识了@Controller，@Service，@Repository，@Component注解的类到spring容器中。如果通过注解实现装配组件，这个配置类需要位于项目的根目录，让Spring明白在哪里扫描。以便扫描到整个项目的组件类 |
| @Import                  | 引入带有@Configuration的java类。                             |
| @ImportResourse          | 引入spring配置文件 applicationContext.xml                    |

注：@Configuration注解的配置类有如下要求：

1. @Configuration不可以是final类型；
2. @Configuration不可以是匿名类；
3. 嵌套的configuration必须是静态类。

### 2）组件注解

| 注解           | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| @Component     | 说明这是一个交给Spring保管的JAVA Bean。泛指各种组件。        |
| @Bean          | 产生一个Bean对象，将它交给spring管理，产生方法只会调用一次。 |
| @Scope         | 声明一个原型（Prototype）的Bean时，需要添加一个额外的`@Scope`注解 |
| @Order         | 指明注入的Bean的顺序                                         |
| @PostConstruct | 定义组件初始化时运行的方法                                   |
| @PreDestroy    | 定义组件销毁前运行的方法                                     |

注：@Bean属性说明

属性有：value、name、autowire、initMethod、destroyMethod。

* name 和 value 两个属性是相同的含义的， 在代码中定义了别名。为 bean 起一个名字，如果默认没有写该属性，那么就使用方法的名称为该 bean 的名称。

* autowire指定 bean 的装配方式， 根据名称 和 根*[欢迎转载听雨的人博客]*据类型 装配， 一般不设置，采用默认即可。autowire指定的装配方式 有三种Autowire.NO (默认设置)、Autowire.BY_NAME、Autowire.BY_TYPE。

* initMethod和destroyMethod指定bean的初始化方法和销毁方法， 直接指定方法名称即可，不用带括号。

### 3）注入注解

| 注解       | 说明                                                 |
| ---------- | ---------------------------------------------------- |
| @Resource  | 按名称注入                                           |
| @Autowired | 按类型注入                                           |
| @Inject    | 按类型注入                                           |
| @Value     | 将常量、配置文件中的值、其他bean的属性值注入到变量中 |

#### @Resource

需要JAVA Bean注入时使用，可以写到字段和setter方法上，选其一即可。Resource默认按照名称自动注入，属性 name 解析为bean的名字，type解析为bean的类型。注入规则如下：

- 如果同时指定了name和type，则从Spring上下文中找到唯一匹配的bean进行装配，找不到则抛出异常。
- 如果指定了name，则从上下文中查找名称（id）匹配的bean进行装配，找不到则抛出异常。
- 如果指定了type，则从上下文中找到类似匹配的唯一bean进行装配，找不到或是找到多个，都会抛出异常。
- 如果既没有指定name，又没有指定type，则自动按照byName方式进行装配；如果没有匹配，则回退为一个原始类型进行匹配，如果匹配则自动装配。

#### @Autowired

需要JAVA Bean注入时使用，可以写到字段和setter方法上，选其一即可。但是`Autowired`只按类型注入，默认情况下要求依赖的对象必须存在，如果允许null值，需要设置属性required=false，如果需要按名称来装配，需要和@Qualifier注解一起使用。

#### @Inject

需要JAVA Bean注入时使用，可以作用在变量、setter方法、构造函数上。默认根据类型type进行自动装配的，如果需要按名称进行装配，则需要配合@Named。

#### @Value

通过注解将常量、配置文件中的值、其他bean的属性值注入到变量中，作为变量的初始值。bean属性、系统属性、表达式注入，使用@Value("#{}")。bean属性注入*[Power By听雨的人]*需要注入者和被注入者属于同一个IOC容器，或者父子IOC容器关系，在同一个作用域内。配置文件属性注入@Value*[Power By听雨的人]*("${}")

### 4）MVC注解

| 注解               | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
| @Controller        | 负责处理由DispatcherServlet 分发的请求，把用户请求的数据经过处理封装成一个模型，然后再把这个模型返回给对应的视图进行展示。Controller 不会直接依赖于HttpServletRequest 和HttpServletResponse 等HttpServlet 对象，它们可以通过Controller 的方法参数灵活的获取到。 |
| @Service           | 修饰MVC中Service层的组件                                     |
| @Repository        | 注解DAO层（Mapper层）                                        |
| @RequestBody       | 修饰返回的数据，当返回的数据不是html标签的页面，而是其他某种格式的数据时（如json、xml等）使用。 |
| @RestController    | 相当于@Controller和@ResponseBody                             |
| @RequestMapping    | 是用来处理请求地址映射的注解，可以用于类或者方法上。用在类上表示类中所有响应请求的方法都是以该地址作为父路径。一共有六个属性。 |
| @RequestParam      | 获取前端请求传来的参数，有三个属性：defaultValue 表示设置默认值，required 通过boolean设置是否是必须要传入的参数，value 值表示接受的传入的参数类型。 |
|                    |                                                              |
| @ModelAttribute    |                                                              |
| @SessionAttributes |                                                              |
| @PathVarible       |                                                              |

注：@RequestMapping的六个属性

1. value：指定请求的实际地址（默认属性）
2. method：指定请求的类型，GET POST DELETE PUT
3. consumes：指定处理请求的Content-Type(内容类型)
4. produces：指定返回的内容类型，仅当请求包含该类型时才回返回相应的数据
5. params：指定请求必须包含某些参数值才会处理该请求
6. headers：指定请求必须包含某些指定的header值才会处理该请求

### 5）AOP切面注解

Spring支持AspectJ的注解式aop编程，需要在java的配置类中使用@EnableAspectJAutoProxy注解开启Spring对AspectJ代理的支持。

| 注解                   | 说明                                                         |
| ---------------------- | ------------------------------------------------------------ |
| @EnableAspectAutoProxy | 开启Aspect代理，使用AOP注解必备                              |
| @Aspect                | 声明一个切面类，该类中的方法都会在合适的时机中插入到需要该方法的地方，方法也需要注解标识 |
| @Before                | 在指定方法执行前执行此方法，需要在注解参数中传入指定方法全名 |
| @After                 | 在指定方法执行后执行此方法                                   |
| @AfterRunning          | 在方法返回结果后执行此方法                                   |
| @AfterThrowing         | 在方法抛出异常后执行此方法                                   |
| @Around                | 围绕着方法执行                                               |
| @PointCut              |                                                              |


