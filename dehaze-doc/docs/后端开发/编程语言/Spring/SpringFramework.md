---
order: 1
---

# Spring Framework

## IoC 容器

### 容器概念

容器是一种可以存放数据的具体数据结构实现。Spring 容器为特定组件对象提供必要支持的软件环境，这些特定组件对象即 Spring
Bean。Spring 容器提供底层服务包括对象的配置、对象整个生命周期管理、组件的生命周期管理、配置和组装服务、AOP（面向切面编程）支持，以及建立在
AOP 基础上的声明式事务服务等。

Tomcat 就是一个 Servlet 容器，底层实现了 TCP 连接、解析 HTTP 协议等复杂服务，我们无需在组件中编写这些复杂的逻辑。

当把一个 Bean 对象交给 Spring 容器管理时，这个对象会被拆解放到 Bean 的定义中，然后由 Spring 统一装配，包括 Bean
的初始化、属性填充等过程。对于 Spring 容器，需要一种可以用于存放对象、可以通过名称索引查找对象的数据结构，通常选择
HashMap。Spring 容器的实现需要对象的定义、注册、获取三个基本步骤。

- 定义：定义存放在 Spring 中的对象，称为 Bean 对象。这个定义过程称为 BeanDefinition，设计为一个类，包含
  singleton、prototype、BeanClassName 等属性
- 注册：将 Bean 对象注册到 Spring 容器中，或称为将对象放入 HashMap 中以便后续获取，Key 为 Bean 对象名称，Value 为对象本身
- 获取：通过 Bean 对象的名称获取该对象

### 容器设计

Spring 中 Bean 对象的创建由容器本身完成，而非在调用时传入已实例化的对象。对于同一类型的对象，有时需要一个实例，有时需要多个实例，因此需要重点考虑单例对象的设计。

### 控制反转 (IoC)

控制反转（Inversion of Control，IoC）指程序中对象的创建、配置等控制权由应用程序转移到 IoC
容器。对于具体实例对象，其所有组件对象不再由应用程序自己创建和配置，而是由 IoC 容器负责，使应用程序能够直接使用已创建并配置好的组件。

在设计上，IoC 容器是无侵入的，应用程序的组件无需实现 Spring 的特定接口，这些组件既可以在 Spring
容器中运行，又能够独立编写代码组装所需对象。在测试时，也不需要实现接口，不依赖 Spring 容器，可单独测试。

### 依赖注入

组件需要通过注入机制装入到实例对象中供其使用。依赖注入方式有两种：

1. 通过 `setXXX()` 方法注入
2. 通过构造方法注入

Spring 的 IoC 容器同时支持属性注入和构造方法注入，并允许混合使用。

由于 IoC 容器需要负责实例化所有组件对象，因此需要告诉容器如何创建组件对象以及各组件对象间的依赖关系，即装配方式。Spring
可通过两种方式实现：

1. XML 配置文件
2. 注解

### 组件装配

#### XML 装配组件

需要将组件间的依赖关系描述出来，然后交给容器创建并装配。

编写配置文件 application.xml，告诉 Spring 容器如何创建并按顺序正确注入到相应组件中。Bean 表示这是一个 Java Bean 或组件。id
唯一标识 Java Bean，class 提供文件路径。每个 Java Bean 内部可以有一个或多个需要注入的属性，以 property 标签表示。这些属性也是
Java Bean，name 表示组件内部需要注入属性的名称，ref 表示需要注入属性指向的 Java Bean 的 id。

总结来说，Java Bean 通过引用注入，数据类型通过 value 注入。

```xml
<bean id="userService" class="com.itranswarp.learnjava.service.UserService">
    <property name="mailService" ref="mailService" />  <!--引用注入-->
    <property name="username" value="root" />          <!--值注入-->
    <property name="password" value="password" />
</bean>
```

在代码中加载配置文件，创建 Spring IoC 容器实例并加载配置文件。Spring 容器命名为应用程序上下文
ApplicationContext，是加载配置文件的接口，有多个实现类。通过 XML 加载需要 ClassPathXmlApplicationContext
实现类自动从项目路径下查找指定配置文件，参数为配置文件名。通过注解加载需要 AnnotationConfigApplicationContext
实现类，参数为配置类名称，必须传入标注了 @Configuration 的类名。

#### 注解装配组件

详见组件详解部分。

## 面向切面编程 (AOP)

实际开发中有很多功能是多个组件通用但非核心的业务逻辑。AOP 将切面（非核心但必要的逻辑）织入核心逻辑中。调用业务方法时，Spring
会对方法进行拦截，并在拦截前后进行安全检查、日志、事务等处理，从而完成整个业务流程。有 3 种实现方式：

- 编译期：由编译器把切面编译进字节码
- 类加载器：当目标装载到 JVM 时，通过特殊类加载器对目标类字节码重新增强
- 运行期：通过动态代理实现运行期动态织入

Spring 的 AOP 实现基于 JVM 的动态代理，通过 AOP 技术可将权限检查、日志、事务等常用功能从每个业务方法中剥离出来。

使用 AOP 需要三步：

1. 定义切入方法，并在方法上通过 AspectJ 注解告诉 Spring 应在何处调用此方法
2. 在需要切入方法的地方标记 @Component 和 @Aspect
3. 在 @Configuration 类上标注 @EnableAspectJAutoProxy

还可通过自定义注解切入功能。在需要切入常用功能的方法头上标记自定义注解，而在切入方法（常用功能逻辑所在方法）的 AOP
注解参数中填入该注解名称，参数格式为 "@annotation(your_annotation_name)"，只要标注了自定义注解的地方，Spring 都会把切入方法切入到里面。
