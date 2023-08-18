# MyBatis 练习

## 基础使用

### 从XML中构建SqlSessionFactory

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
  	<mapper resource="cn.pei.mybatis.UserMapper.xml"
  </mappers>
</configuration>
```

```java
String resource = "cn/pei/mybatis/mybatis-config.xml";
InputStream inputStream = Resources.getResourceAsStream(resource);
SqlSessionFactory sqlSessionFactory = new SqlSessionFactoryBuilder().build(inputStream);
try (SqlSession session = sqlSessionFactory.openSession()){
  BlogMappe mapper = session.getMapper(BlogMapper.class);
  Blog blog = mapper.selectBlog(101);
}
```

在mybatis中，一个Sql语句可以通过Java语句定义，也可以通过xml定义。

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

在一个xml映射文件中，存在有mapper元素，命名空间为org.mybatis.example.BlogMapper，这样该名字可以直接映射到在命名空间中同名的映射器类，并且将已经映射的select语句匹配到对应名参数和返回类型的方法。

### 通过Java代码构建SqlSessionFactory

## 作用域和生命周期



## SQL语句映射



## MyBatis配置
### 基于XML对MyBatis进行配置



### 采用SpringBoot自动配置



### 基于XML的SQL语句映射文件



## 动态SQL



## Java API



## SQL语句构建器

