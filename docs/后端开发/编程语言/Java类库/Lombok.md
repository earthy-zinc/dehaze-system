

# Lombok

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
