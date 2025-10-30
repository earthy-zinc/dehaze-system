# MyBatis

## 概述

MyBatis 是一个优秀的持久层框架，其功能架构分为三层：

1. 接口层：提供给程序员调用的接口 API，程序员通过这些 API 来操作数据库，接口层收到调用请求就会转而调用数据处理层来完成具体的数据处理。
2. 数据处理层：负责对具体的从参数映射、SQL 语句的解析、执行、以及执行结果的映射处理等，主要的作用就是根据调用的请求完成一次数据库的操作。
3. 基础支撑层：负责最基础的功能支撑，包括连接池管理、事务管理、配置的加载、缓存处理。这些都是进行数据库查询通用的东西，我们把他从数据库操作中抽取出来作为最基本的组件，为上层的数据处理层提供最基础的支撑。

引导层不参与 SQL 语句的处理，它只是为了在应用程序中配置 MyBatis 的各种功能如何运行。

## 总体框架设计

### 接口层

接口层是 MyBatis 提供给程序员调用的应用程序接口，通过接口层可以方便地调用数据库获取数据。MyBatis 提供两种方式与数据库进行交互：

- 创建 SqlSession 对象：创建一个 SQL 语句会话对象（SqlSession）使用该对象完成对数据库的交互，该对象内部维护了与数据库的连接，提供了与数据库进行增删改查操作的方法。
- 使用 Mapper 接口：通过接口调用的方式，MyBatis 中应用程序的某个对象与数据库某个表连接的桥梁是通过 mapper 映射实现的，配置文件中的每一个
  mapper 结点都对应着一个 mapper 接口。接口中的每一个方法对应配置文件中每一条 SQL 语句。配置好以后，MyBatis
  会根据相应的接口方法通过动态代理生成一个 mapper 实例，调用 mapper 接口的某一个方法时，MyBatis 会根据这个方法的名称还有参数类型，确定
  SQL 语句，实现对数据库的操作。

### 数据处理层

在 Java 中调用对数据库增删改查方法时，会传入一些参数，这些参数可能是具体要存的数据或其他内容，MyBatis
的数据处理层所要实现的功能就是从这里展开的，主要完成两个功能：

- 通过传入的参数构建动态的 SQL 语句
- 执行对应的 SQL 语句并封装查询结果映射到 Java 对象中

### 框架支撑层

框架支撑层负责数据库查询中的一些通用功能，主要包括：

- 事务管理机制
- 连接池管理机制
- 缓存机制：为了减少数据库的压力，提高数据利用率，MyBatis 会对一些查询结果缓存到本地中，在一定的时间间隔内多次的同一查询，MyBatis
  会直接返回缓存的内容，而不会再去数据库中查找
- SQL 语句的配置方式：Java 程序中 SQL 语句配置方式有两种，一种是通过 XML 来配置，另一种是在 mapper
  接口上使用注解来配置。这个功能就是为了识别并处理两种不同的配置方式而存在的

## 功能实现层次结构

1. 使用 MyBatis 查询数据库，首先需要创建一个 SQL 会话对象，也就是 SqlSession，创建完成之后，就开启了一个与数据库的连接会话，可以通过这个对象来执行
   SQL 语句、提交或者回滚事务
2. 但实际上出于分离职责、防止一个对象身兼太多职责，SqlSession 只是执行数据库查询的第一层对象，它会紧接着调用 Executor
   对象，这个对象会负责 SQL 动态语句的生成，对查询出来的结果进行缓存，对这些结果进行维护，定期删除等
3. 其次是语句处理器对象，由于 Java 程序在底层与数据库的交互是通过 JDBC 实现的，MyBatis 是在 JDBC 的基础上做出了进一步的封装。因此语句处理器对象
   StatementHandler 主要负责与 JDBC 语句之间的交互，设置语句参数，将返回的结果映射到 Java 对象
4. 接下来就是 JDBC 层，是真正负责查询数据库的部分

## MyBatis 初始化

MyBatis 初始化的方式主要有两种：通过 XML 配置和基于 Java 的 API。

### XML 配置初始化

从 XML 中构建 SqlSessionFactory：

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE configuration
  PUBLIC "-//mybatis.org//DTD Config 3.0//EN"
  "https://mybatis.org/dtd/mybatis-3-config.dtd">
<configuration>
	<environments default="development">
  	<environment id="development">
    	<transactionManager type="JDBC"/>
      <dataSource type="POOLED">
      	<property name="driver" value="${driver}"/>
        <property name="url" value="${url}"/>
        <property name="username" value="${username}"/>
        <property name="password" value="${password}"/>
      </dataSource>
    </environment>
  </environments>
  <mappers>
  	<mapper resource="cn.pei.mybatis.UserMapper.xml"/>
  </mappers>
</configuration>
```

```java
String resource = "cn/pei/mybatis/mybatis-config.xml";
InputStream inputStream = Resources.getResourceAsStream(resource);
SqlSessionFactory sqlSessionFactory = new SqlSessionFactoryBuilder().build(inputStream);
try (SqlSession session = sqlSessionFactory.openSession()){
  BlogMapper mapper = session.getMapper(BlogMapper.class);
  Blog blog = mapper.selectBlog(101);
}
```

在 MyBatis 中，一个 SQL 语句可以通过 Java 语句定义，也可以通过 XML 定义：

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE mapper
  PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
  "https://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="org.mybatis.example.BlogMapper">
  <select id="selectBlog" resultType="Blog">
    select * from Blog where id = #{id}
  </select>
</mapper>
```

在一个 XML 映射文件中，存在有 mapper 元素，命名空间为 org.mybatis.example.BlogMapper，这样该名字可以直接映射到在命名空间中同名的映射器类，并且将已经映射的
select 语句匹配到对应名参数和返回类型的方法。

### Java API 初始化

（待补充）

## 作用域和生命周期

（待补充）

## SQL 语句映射

（待补充）

## MyBatis 配置

### 基于 XML 对 MyBatis 进行配置

（待补充）

### 采用 SpringBoot 自动配置

（待补充）

### 基于 XML 的 SQL 语句映射文件

（待补充）

## 动态 SQL

（待补充）

## Java API

（待补充）

## SQL 语句构建器

（待补充）
