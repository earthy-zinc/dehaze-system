# Lombok

Lombok 是一个 Java 库，通过注解的形式自动插入编辑器和构建工具中，以减少 Java 开发人员必须编写的样板代码量。

## 常用注解

| 注解                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| @Slf4j                   | 自动生成该类的 log 静态常量                                  |
| @Log4j2                  | 注解在类上。为类提供一个属性名为 log 的 log4j 日志对象，和@Log4j 注解类似。 |
| @Setter                  | 注解在属性上，为属性提供 setter 方法。注解在类上，为所有属性添加 setter 方法 |
| @Getter                  | 注解在属性上，为属性提供 getter 方法。注解在类上，为所有属性添加 getter 方法 |
| @EqualsAndHashCode       | 生成 equals(Object other) 和 hashCode() 方法               |
| @RequiredArgsConstructor | 生成包含所有带 @NonNull 注解或 final 修饰的成员变量的构造函数 |
| @NoArgsConstructor       | 生成无参构造函数                                             |
| @AllArgsConstructor      | 生成包含所有属性的构造函数                                   |
| @NotNull                 | 用于方法参数或返回值，表示不能为空                           |
| @NullAble                | 用于方法参数或返回值，表示可以为空                           |
| @ToString                | 生成 toString() 方法                                         |
| @Value                   | 所有变量为 final，等同于添加@Getter @ToString @EqualsAndHashCode @RequiredArgsConstructor |
| @Data                    | 等同于添加@Getter/@Setter @ToString @EqualsAndHashCode @RequiredArgsConstructor |
| @Builder                 | 自动生成流式 set 值写法                                      |

注：@EqualsAndHashCode 默认情况下，会使用所有非瞬态(non-transient)和非静态(non-static)字段来生成 equals 和 hascode 方法，也可以指定具体使用哪些属性。如果某些变量不想加入判断通过 exclude 排除，或者使用 of 指定使用某些字段