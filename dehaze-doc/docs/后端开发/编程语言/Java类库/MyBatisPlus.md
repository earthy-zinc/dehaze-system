# MyBatis Plus

## 项目集成

在项目中引入 MyBatis Plus 需要以下步骤：

1. 添加相应依赖
2. 在 Spring Boot 启动类中添加 `@MapperScan` 注解，扫描 Mapper 文件夹
3. 编写 Mapper 包下的接口，继承 MyBatis Plus 提供的 `BaseMapper<T>`

### 默认映射关系

MyBatis Plus 实体类默认具有以下映射关系：

| 映射关系                       | 说明                                                      |
| ------------------------------ | --------------------------------------------------------- |
| 下划线映射为实体类的驼峰式命名 | 表名：st_user → stUser 类。字段名：real_name → realName   |
| 数据表主键名为 id              | 插入数据时会自增，不需要手动指定                          |
| 字段名与实体类属性一一对应     |                                                           |

### 常用注解

| 注解        | 说明                         |
| ----------- | ---------------------------- |
| @TableName  | 表名注解，标识实体类对应的表 |
| @TableId    | 主键注解，用于实体类主键字段 |
| @TableField | 非主键的字段注解             |

### 常见配置

```yaml
mybatis-plus:
  # 该配置项可以在编写 mapper.xml 文件时省略 resultType 的全类名，直接使用相应包下的类名即可
  type-aliases-package: com.itheima.mp.domain.po
  global-config:
    db-config:
      logic-delete-field: deleted # 配置逻辑删除字段
  configuration:
    # 配置枚举类与 POJO 类相关属性及数据库相应数据之间的映射关系
    default-enum-type-handler: com.baomidou.mybatisplus.core.handlers.MybatisEnumTypeHandler
```

## 核心功能

### 条件构造器

#### QueryWrapper

无论是修改、删除还是查询，都可以使用 QueryWrapper 来构建查询条件。

查询示例：查询名字中带 `o` 且存款大于等于 1000 元的用户：

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

更新示例：更新用户名为 jack 的用户余额为 2000：

```java
@Test
void testUpdateByQueryWrapper() {
    // 1.构建查询条件 where name = "Jack"
    QueryWrapper<User> wrapper = new QueryWrapper<User>().eq("username", "Jack");
    // 2.更新数据，user 中非 null 字段都会作为 set 语句
    User user = new User();
    user.setBalance(2000);
    userMapper.update(user, wrapper);
}
```

#### UpdateWrapper

基于 BaseMapper 中的 update 方法更新时只能直接赋值，对于一些复杂需求就难以实现。

例如：更新 id 为 `1,2,4` 的用户余额，扣 200，对应的 SQL 应该是：

```sql
UPDATE user SET balance = balance - 200 WHERE id in (1, 2, 4)
```

SET 的赋值结果是基于字段现有值的，这时可以利用 UpdateWrapper 中的 setSql 功能：

```java
@Test
void testUpdateWrapper() {
    List<Long> ids = List.of(1L, 2L, 4L);
    // 1.生成SQL
    UpdateWrapper<User> wrapper = new UpdateWrapper<User>()
            .setSql("balance = balance - 200") // SET balance = balance - 200
            .in("id", ids); // WHERE id in (1, 2, 4)
    // 2.更新，注意第一个参数可以给 null，也就是不填更新字段和数据，
    // 而是基于 UpdateWrapper 中的 setSQL 来更新
    userMapper.update(null, wrapper);
}
```

#### LambdaQueryWrapper 和 LambdaUpdateWrapper

QueryWrapper 和 UpdateWrapper 在构造条件时都需要写死字段名称，会出现字符串"魔法值"，这在编程规范中不推荐。

可以基于变量的 getter 方法结合反射技术，将条件对应字段的 getter 方法传递给 MyBatis Plus，它就能计算出对应的变量名。传递方法可以使用 JDK8 中的方法引用和 Lambda 表达式。

MyBatis Plus 提供了一套基于 Lambda 的 Wrapper，包含两个：

- LambdaQueryWrapper
- LambdaUpdateWrapper

分别对应 QueryWrapper 和 UpdateWrapper：

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

### 自定义 SQL

SQL 语句最好都维护在持久层，而不是业务层。对于 UpdateWrapper 案例，由于条件是 in 语句，只能将 SQL 写在 Mapper.xml 文件，利用 foreach 来生成动态 SQL。MyBatis Plus 提供了自定义 SQL 功能，可以利用 Wrapper 生成查询条件，再结合 Mapper.xml 编写 SQL。

#### 基本用法

对于上述 UpdateWrapper 案例，可以改为如下写法：

```java
// 模拟业务层（service 层）调用 mapper
@Test
void testCustomWrapper() {
    // 1.准备自定义查询条件
    List<Long> ids = List.of(1L, 2L, 4L);
    QueryWrapper<User> wrapper = new QueryWrapper<User>().in("id", ids);

    // 2.调用 mapper 的自定义方法，直接传递 Wrapper
    userMapper.deductBalanceByIds(200, wrapper);
}
```

然后在 mapper 中自定义 SQL，其中 @Param("ew") 为必写项，并且只能写 `ew`：

```java
public interface UserMapper extends BaseMapper<User> {
    @Select("UPDATE user SET balance = balance - #{money} ${ew.customSqlSegment}")
    void deductBalanceByIds(@Param("money") int money, @Param("ew") QueryWrapper<User> wrapper);
}
```

#### 多表关联

理论上 MyBatis Plus 不支持多表查询，但可以利用 Wrapper 中自定义条件结合自定义 SQL 来实现多表查询效果：

```java
// 模拟业务层（service 层）调用 mapper
@Test
void testCustomJoinWrapper() {
    // 1.准备自定义查询条件
    QueryWrapper<User> wrapper = new QueryWrapper<User>()
            .in("u.id", List.of(1L, 2L, 4L))
            .eq("a.city", "北京");

    // 2.调用 mapper 的自定义方法
    List<User> users = userMapper.queryUserByWrapper(wrapper);

    users.forEach(System.out::println);
}
```

在 UserMapper 中自定义方法：

```java
@Select("SELECT u.* FROM user u INNER JOIN address a ON u.id = a.user_id ${ew.customSqlSegment}")
List<User> queryUserByWrapper(@Param("ew")QueryWrapper<User> wrapper);
```

## 扩展功能

### 静态工具(Db 类)

有时 Service 之间会相互调用，为避免循环依赖问题，MyBatis Plus 提供了一个静态工具类 `Db`，其中的一些静态方法与 `IService` 中方法签名基本一致，也可以帮助实现 CRUD 功能。

需求：根据 id 查询用户，同时返回用户收货地址列表：

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
    // 3.处理 vo
    UserVO userVO = BeanUtil.copyProperties(user, UserVO.class);
    userVO.setAddresses(BeanUtil.copyToList(addresses, AddressVO.class));
    return userVO;
}
```

在查询地址时，采用了 Db 的静态方法，因此避免了注入 AddressService，减少了循环依赖的风险。

使用 Db 类的注意事项：只有在较新版本的 MyBatis Plus 中才支持该类，如 `3.5.3.1`。

### 通用枚举

User 类中有一个用户状态字段：

```java
private Integer status; // 表示使用状态，1 正常 2 冻结
```

像这种字段一般会定义一个枚举，做业务判断时就可以直接基于枚举做比较。但数据库采用的是 `int` 类型，对应的 PO 也是 `Integer`。因此业务操作时必须手动把枚举与 Integer 转换，非常麻烦。

MyBatis Plus 提供了一个处理枚举的类型转换器，可以把枚举类型与数据库类型自动转换。

#### 定义枚举

1. 首先定义一个用户状态的枚举类：

```java
package com.xxx.xx.enums;

import com.baomidou.mybatisplus.annotation.EnumValue;
import lombok.Getter;

@Getter
public enum UserStatus {
    NORMAL(1, "正常"),
    FREEZE(2, "冻结");
    
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

2. 然后把 User 类中的 status 字段改为 UserStatus 类型：

```java
private UserStatus status; // 表示使用状态，1 正常 2 冻结
```

3. 要让 MyBatis Plus 处理枚举与数据库类型自动转换，必须告诉 MyBatis Plus 枚举中哪个字段的值作为数据库值。MyBatis Plus 提供了 @EnumValue 注解来标记枚举属性，因此要在 value 属性上添加 @EnumValue 注解。

#### 配置枚举处理器

在 application.yaml 文件中添加配置：

```yaml
mybatis-plus:
  configuration:
    default-enum-type-handler: com.baomidou.mybatisplus.core.handlers.MybatisEnumTypeHandler
```

#### 测试

```java
@Test
void testService() {
    List<User> list = userService.list();
    list.forEach(System.out::println);
}
```

返回的结果中，status 字段值为：`status="NORMAL"`，但将该值返回给前端并不见名知意，因此为了保证格式的统一，需要在 UserStatus 枚举中通过 `@JsonValue` 注解标记 JSON 序列化时展示的字段。

此时再次测试会发现 `status="正常"`，达到了预期效果。

## 插件功能

### 分页插件

在未引入分页插件的情况下，MyBatis Plus 不支持分页功能，IService 和 BaseMapper 中的分页方法都无法正常起效。因此，必须要配置分页插件。

#### 配置分页插件

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

#### 分页 API

```java
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

这里用到了分页参数 Page，即可以支持分页参数，也可以支持排序参数。常见的 API 如下：

```java
int pageNo = 1, pageSize = 5;
// 分页参数
Page<User> page = Page.of(pageNo, pageSize);
// 排序参数, 通过 OrderItem 来指定
page.addOrder(new OrderItem("balance", false));

userService.page(page);
```

### 通用分页实体

#### PageQuery 实体

```java
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

    public <T> Page<T> toMpPage(OrderItem ... orders){
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

#### PageDTO 实体

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

如果希望自定义 PO 到 VO 的转换过程，可以这样做：

```java
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