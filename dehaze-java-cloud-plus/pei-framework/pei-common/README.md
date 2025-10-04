`pei-common` 是一个通用工具模块，主要作用是为整个项目提供基础功能支持。它包含了 **业务常量、异常处理、数据结构、工具类**
等核心内容，贯穿于整个微服务架构的各个层级（如 Controller、Service、Mapper），确保代码的 **可维护性、复用性、规范性**。

---

## 一、模块概述

### ✅ 模块定位

- **基础工具库**：封装了常用的工具方法。
- **统一错误码**：定义全局和业务错误码，提升系统健壮性。
- **通用 POJO 类**：如 `CommonResult`, `PageParam`, `PageResult`，用于统一接口返回格式。
- **枚举与校验**：标准化状态码、用户类型等枚举，并提供自定义校验注解。
- **业务通用模型**：如 `KeyValue`、`ArrayValuable`，便于在不同层之间传递数据。

---

## 二、包结构详解

```text
src/main/java/
└── com/pei/dehaze/framework/common/
    ├── core/             // 核心类，如 KeyValue、ArrayValuable
    ├── enums/            // 枚举类，如 UserTypeEnum、CommonStatusEnum
    ├── exception/        // 异常体系，包括 ErrorCode、ServerException、ServiceException
    ├── pojo/             // 数据传输对象（DTO），如 CommonResult、PageParam
    ├── util/             // 工具类，涵盖缓存、集合、日期、JSON、HTTP、Spring、IO 等
    └── validation/       // 自定义验证器，如 @Mobile, @InEnum
```

---

## 三、关键包说明与实现原理

### 1️⃣ `core` 包

#### 🔹 `ArrayValuable.java`

```java
public interface ArrayValuable {
    Integer getValue();
}
```

- **作用**：用于表示数组中的每个元素都可以被转换为整数值（如状态码）。
- **使用场景**：在枚举中继承该接口，便于统一获取值。

#### 🔹 `KeyValue.java`

```java

@Data
@NoArgsConstructor
@AllArgsConstructor
public class KeyValue<K, V> {
    private K key;
    private V value;
}
```

- **作用**：泛型键值对，用于构建字典、选项列表等。
- **示例**：

  ```java
  List<KeyValue<String, String>> options = Arrays.asList(
      new KeyValue<>("1", "男"),
      new KeyValue<>("2", "女")
  );
  ```

---

### 2️⃣ `enums` 包

#### 🔹 `UserTypeEnum.java`

```java

@Getter
public enum UserTypeEnum implements ArrayValuable {
    MEMBER(1, "会员"),
    ADMIN(2, "管理员");

    private final Integer value;
    private final String label;
}
```

- **作用**：定义用户类型枚举，结合 `@InEnum` 注解进行参数校验。
- **优点**：避免魔法数字，增强可读性和安全性。

#### 🔹 `CommonStatusEnum.java`

```java

@Getter
public enum CommonStatusEnum implements ArrayValuable {
    ENABLE(0, "启用"),
    DISABLE(1, "禁用");

    private final Integer value;
    private final String label;
}
```

- **作用**：用于表示通用状态（如是否启用、上下架等）。
- **应用**：在数据库字段、接口参数中广泛使用。

---

### 3️⃣ `exception` 包

#### 🔹 `ErrorCode.java`

```java
public interface ErrorCode {
    Integer getCode();

    String getMsg();
}
```

- **作用**：定义错误码接口，所有具体错误码实现该接口。

#### 🔹 `GlobalErrorCodeConstants.java`

```java
public interface GlobalErrorCodeConstants {
    ErrorCode SUCCESS = new ErrorCodeImpl(200, "成功");
    ErrorCode INTERNAL_SERVER_ERROR = new ErrorCodeImpl(500, "服务器内部异常");
}
```

- **作用**：定义全局错误码，供异常处理器统一返回。

#### 🔹 `ServiceException.java`

```java
public class ServiceException extends RuntimeException implements HasErrorCode {
    private final ErrorCode errorCode;

    public ServiceException(ErrorCode errorCode) {
        super(errorCode.getMsg());
        this.errorCode = errorCode;
    }
}
```

- **作用**：业务逻辑抛出异常时使用，包含错误码和描述信息。

---

### 4️⃣ `pojo` 包

#### 🔹 `CommonResult.java`

```java

@Data
public class CommonResult<T> implements Serializable {
    private Integer code;
    private T data;
    private String msg;

    public static <T> CommonResult<T> success(T data) {
        CommonResult<T> result = new CommonResult<>();
        result.code = GlobalErrorCodeConstants.SUCCESS.getCode();
        result.data = data;
        return result;
    }

    public static <T> CommonResult<T> error(Integer code, String message) {
        CommonResult<T> result = new CommonResult<>();
        result.code = code;
        result.msg = message;
        return result;
    }
}
```

- **作用**：统一 API 返回格式，前后端交互标准。
- **优点**：
  - 明确区分成功/失败状态。
  - 支持泛型数据返回。
  - 提供 `isSuccess()` 方法判断结果。

#### 🔹 `PageParam.java`

```java

@Data
public class PageParam {
    @NotNull(message = "pageNum 不能为空")
    private final Integer pageNum = 1;
    @NotNull(message = "pageSize 不能为空")
    private final Integer pageSize = 10;
}
```

- **作用**：分页请求参数，用于 Controller 接口接收。
- **扩展性**：可继承或组合扩展其他查询条件。

#### 🔹 `PageResult.java`

```java

@Data
public class PageResult<T> {
    private List<T> list;
    private Long total;
}
```

- **作用**：分页响应数据结构，Controller 层返回给前端。
- **搭配使用**：通常与 `PageParam` 配合使用，实现分页查询。

---

### 5️⃣ `util` 包

#### 📁 `collection` 子包

- **作用**：集合工具类，如 `CollectionUtils.filterList(...)`。
- **示例**：

  ```java
  List<User> activeUsers = CollectionUtils.filterList(users, user -> user.getStatus() == 1);
  ```

#### 📁 `date` 子包

- **作用**：时间处理工具类，如 `LocalDateTimeUtil`。
- **优势**：兼容 Java 8 的 `LocalDateTime`，解决格式化问题。

#### 📁 `json` 子包

- **作用**：封装 JSON 序列化/反序列化逻辑。
- **关键类**：
  - `JsonUtils.toJsonString(...)`
  - `JsonUtils.parseObject(...)`

#### 📁 `validation` 子包

- **作用**：自定义校验注解及其实现。
- **常用注解**：
  - `@Mobile`：手机号校验。
  - `@InEnum`：限定输入必须是某个枚举值。

---

### 6️⃣ `validation` 包

#### 🔹 `InEnum.java`

```java

@Target({ElementType.FIELD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = InEnumValidator.class)
public @interface InEnum {
    Class<? extends Enum<?>> value();

    String message() default "不在指定范围内";
}
```

- **作用**：限制字段值必须属于指定枚举。
- **使用方式**：

  ```java
  @InEnum(UserTypeEnum.class)
  private Integer userType;
  ```

#### 🔹 `Mobile.java`

```java

@Constraint(validatedBy = MobileValidator.class)
public @interface Mobile {
    String message() default "手机号不合法";
}
```

- **作用**：校验手机号格式。
- **使用方式**：

  ```java
  @Mobile
  private String mobile;
  ```

---

## 四、关键类实现细节

### 1️⃣ `NumberUtils.java`

```java
public class NumberUtils {
    public static Long parseLong(String str) {
        return StrUtil.isNotEmpty(str) ? Long.valueOf(str) : null;
    }

    public static boolean isAllNumber(List<String> values) {
        if (CollUtil.isEmpty(values)) return false;
        for (String value : values) {
            if (!NumberUtil.isNumber(value)) return false;
        }
        return true;
    }

    public static double getDistance(double lat1, double lng1, double lat2, double lng2) {
        double radLat1 = lat1 * Math.PI / 180.0;
        double radLat2 = lat2 * Math.PI / 180.0;
        double a = radLat1 - radLat2;
        double b = lng1 * Math.PI / 180.0 - lng2 * Math.PI / 180.0;
        double distance = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin(a / 2), 2)
                + Math.cos(radLat1) * Math.cos(radLat2) * Math.pow(Math.sin(b / 2), 2)));
        return Math.round(distance * 6378.137 * 10000d) / 10000d;
    }
}
```

- **功能**：
  - 数值解析（null 安全）。
  - 批量字符串是否都为数字。
  - 计算两个经纬度之间的地球距离（单位 km）。
- **适用场景**：
  - 地理围栏、位置排序、距离筛选等。

---

## 五、总结

| 包名           | 主要职责   | 关键类                                           |
|--------------|--------|-----------------------------------------------|
| `core`       | 核心数据结构 | `KeyValue`、`ArrayValuable`                    |
| `enums`      | 枚举定义   | `UserTypeEnum`、`CommonStatusEnum`             |
| `exception`  | 异常处理   | `GlobalErrorCodeConstants`、`ServiceException` |
| `pojo`       | 数据传输对象 | `CommonResult`、`PageParam`                    |
| `util`       | 各种工具类  | `JsonUtils`、`NumberUtils`                     |
| `validation` | 自定义校验  | `InEnum`、`Mobile`                             |
