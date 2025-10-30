`pei-spring-boot-starter-excel` 是一个 **Excel 拓展模块（Excel Extension Module）**，其核心作用是为企业级应用提供统一的
Excel 读写能力，并支持以下功能：

- 字典格式化与反向解析
- 下拉选择列自动生成
- 地区、金额等特定字段类型转换
- JSON 序列化/反序列化支持
- 高性能缓存字典数据以提升转换效率

该模块基于 `EasyExcel + Apache POI` 实现高性能 Excel 处理，并结合系统字典模块实现数据自动映射，适用于电商、CRM、ERP、AI
等需要处理大量 Excel 数据导入导出的场景。

---

## ✅ 模块概述

### 🎯 模块定位

- **目标**：构建统一的 Excel 封装层，支持：
    - 自动将数据库中的字典值转换为前端可读标签（如 `1 -> "启用"`）
    - 支持下拉框列生成（方便用户选择固定选项）
    - 支持地区、金额、JSON 等复杂类型字段的转换
- **应用场景**：
    - 管理后台 Excel 导入/导出
    - 用户 APP 批量操作
    - 报表系统数据导出

### 🧩 技术栈依赖

- **Excel 引擎**：`Alibaba EasyExcel`
- **字典服务**：`DictFrameworkUtils` + `DictDataCommonApi`
- **数据转换**：`Converter` 接口扩展
- **Spring Boot 3.4 + Java 17**

---

## 📁 目录结构说明

```
src/main/java/
└── com/pei/dehaze/framework/excel/
    ├── core/                    // 核心逻辑实现
    │   ├── annotations/         // 注解定义
    │   ├── convert/             // 类型转换器
    │   ├── function/            // Excel 列下拉数据源接口
    │   ├── handler/             // 写入处理器（如下拉框生成）
    │   └── util/                // 工具类封装
    └── dict/                    // 字典相关拓展
        ├── config/              // 字典配置类
        └── core/                // 字典工具类
```

---

## 🔍 关键包详解

### 1️⃣ `annotations` 包 —— Excel 注解定义

#### 示例：`DictFormat.java`

```java
@Target({ElementType.FIELD})
@Retention(RetentionPolicy.RUNTIME)
@Inherited
public @interface DictFormat {
    String value(); // 字典类型，如 SysDictTypeConstants.USER_TYPE
}
```

- **作用**：用于标记字段应使用哪个字典类型进行格式化。
- **使用方式**：
  ```java
  @ExcelProperty("用户状态")
  @DictFormat(SysDictTypeConstants.USER_STATUS)
  private Integer status;
  ```

---

### 2️⃣ `convert` 包 —— Excel 字段类型转换器

#### 示例：`DictConvert.java`

```java
@Override
public Object convertToJavaData(ReadCellData readCellData, ExcelContentProperty contentProperty,
                                GlobalConfiguration globalConfiguration) {
    String type = getType(contentProperty);
    String label = readCellData.getStringValue();
    return DictFrameworkUtils.parseDictDataValue(type, label);
}

@Override
public WriteCellData<String> convertToExcelData(Object object, ExcelContentProperty contentProperty,
                                                GlobalConfiguration globalConfiguration) {
    String type = getType(contentProperty);
    String value = String.valueOf(object);
    return new WriteCellData<>(DictFrameworkUtils.parseDictDataLabel(type, value));
}
```

- **作用**：在读取和写入 Excel 时自动进行字典转换。
- **关键逻辑**：
    - `convertToJavaData`: 将 Excel 中的中文标签（如“启用”）转为数字（如 `1`）
    - `convertToExcelData`: 将数字（如 `1`）转为中文标签（如“启用”）
- **设计模式**：
    - 模板方法模式（统一转换流程）
    - 责任链模式（多个 Converter 可组合）

---

#### 示例：`AreaConvert.java`

```java
public class AreaConvert implements Converter<Object> {
    public Object convertToJavaData(...) {
        String label = readCellData.getStringValue();
        Area area = AreaUtils.parseArea(label);
        return Convert.convert(fieldClazz, area.getId());
    }
}
```

- **作用**：地区字段的 Excel 转换器。
- **关键逻辑**：
    - 使用 `AreaUtils.parseArea(...)` 解析地区名称（如“浙江省杭州市西湖区”）
    - 转换为地区 ID（如 `330105`）
- **适用场景**：
    - 地区字段的 Excel 导入导出

---

#### 示例：`MoneyConvert.java`

```java
public class MoneyConvert implements Converter<Integer> {
    public WriteCellData<String> convertToExcelData(Integer value, ...) {
        BigDecimal result = BigDecimal.valueOf(value).divide(new BigDecimal(100), 2, RoundingMode.HALF_UP);
        return new WriteCellData<>(result.toString());
    }
}
```

- **作用**：金额字段的 Excel 转换器（单位分 → 元）。
- **关键逻辑**：
    - 将数据库中存储的“分”转换为“元”
    - 如 `100` → `"1.00"`
- **优势**：
    - 提供标准化金额显示
    - 避免手动计算精度问题

---

### 3️⃣ `function` 包 —— Excel 列下拉数据源接口

#### 示例：`ExcelColumnSelectFunction.java`

```java
public interface ExcelColumnSelectFunction {
    String getName();
    List<String> getOptions();
}
```

- **作用**：为非字典类型的列提供下拉数据源。
- **使用方式**：
  ```java
  @Component
  public class UserStatusSelectFunction implements ExcelColumnSelectFunction {

      @Override
      public String getName() {
          return "user.status.options";
      }

      @Override
      public List<String> getOptions() {
          return List.of("启用", "禁用");
      }

  }
  ```

---

### 4️⃣ `handler` 包 —— Excel 写入处理器

#### 示例：`SelectSheetWriteHandler.java`

```java
private final Map<Integer, List<String>> selectMap = new HashMap<>();

public SelectSheetWriteHandler(Class<?> head) {
    for (Field field : head.getDeclaredFields()) {
        if (field.isAnnotationPresent(ExcelColumnSelect.class)) {
            getSelectDataList(colIndex, field);
        }
    }
}

private void setColumnSelect(...) {
    CellRangeAddressList rangeAddressList = new CellRangeAddressList(FIRST_ROW, LAST_ROW, colIndex, colIndex);
    DataValidation validation = helper.createValidation(constraint, rangeAddressList);
    writeSheetHolder.getSheet().addValidationData(validation);
}
```

- **作用**：为 Excel 添加下拉选择列。
- **关键逻辑**：
    - 读取字段上的 `@ExcelColumnSelect` 注解
    - 创建独立的 `Sheet` 存储下拉数据
    - 设置单元格下拉约束（Apache POI API）
- **设计模式**：
    - 观察者模式（监听字段变化）
    - 单例模式（共享 Sheet）

---

### 5️⃣ `util` 包 —— Excel 工具类封装

#### 示例：`ExcelUtils.java`

```java
public static <T> void write(HttpServletResponse response, String filename, String sheetName,
                             Class<T> head, List<T> data) throws IOException {
    EasyExcel.write(response.getOutputStream(), head)
            .registerWriteHandler(new LongestMatchColumnWidthStyleStrategy())
            .registerWriteHandler(new SelectSheetWriteHandler(head))
            .sheet(sheetName).doWrite(data);
}
```

- **作用**：封装 Excel 导出常用方法。
- **关键逻辑**：
    - 支持设置响应头、编码、文件名
    - 自动注册 `SelectSheetWriteHandler` 实现下拉框
    - 自动注册 `LongStringConverter` 防止 Long 类型丢失精度
- **优势**：
    - 统一导出格式
    - 提升开发效率

---

### 6️⃣ `dict.core` 包 —— 字典工具类

#### 示例：`DictFrameworkUtils.java`

```java
private static final LoadingCache<String, List<DictDataRespDTO>> GET_DICT_DATA_CACHE = CacheUtils.buildAsyncReloadingCache(
        Duration.ofMinutes(1L),
        new CacheLoader<>() {
            @Override
            public List<DictDataRespDTO> load(String dictType) {
                return dictDataApi.getDictDataList(dictType).getCheckedData();
            }
        });

public static String parseDictDataLabel(String dictType, Integer value) {
    return parseDictDataLabel(dictType, String.valueOf(value));
}

public static String parseDictDataLabel(String dictType, String value) {
    List<DictDataRespDTO> dictDatas = GET_DICT_DATA_CACHE.get(dictType);
    DictDataRespDTO dictData = CollUtil.findOne(dictDatas, data -> Objects.equals(data.getValue(), value));
    return dictData != null ? dictData.getLabel() : null;
}
```

- **作用**：缓存并转换字典数据。
- **关键逻辑**：
    - 使用 Guava 缓存提升性能
    - 支持根据字典类型查询所有标签
    - 支持通过值获取标签、通过标签获取值
- **优势**：
    - 减少对远程服务的频繁调用
    - 提升 Excel 转换效率

---

## 🧠 模块工作流程图解

### 1️⃣ Excel 导出流程

```mermaid
graph TD
    A[Controller 层调用 ExcelUtils.write(...)] --> B[创建 HttpServletResponse 输出流]
    B --> C[加载 DictFrameworkUtils 缓存字典]
    C --> D[遍历字段注解]
    D --> E{是否存在 @ExcelColumnSelect?}
    E -- 是 --> F[创建 SelectSheetWriteHandler]
    E -- 否 --> G[直接写入普通字段]
    F --> H[注册到 Excel 写入器]
    H --> I[生成下拉列表]
    I --> J[输出 Excel 文件]
```

### 2️⃣ Excel 导入流程

```mermaid
graph TD
    A[Controller 层调用 ExcelUtils.read(...)] --> B[解析 Excel 文件]
    B --> C{是否有 @DictFormat 注解?}
    C -- 是 --> D[调用 DictConvert 转换字典值]
    C -- 否 --> E[调用默认转换器]
    D --> F[返回实体对象列表]
```

---

## 🧱 模块间关系图

```mermaid
graph TD
    A[Controller] --> B[ExcelUtils.write(...)]
    B --> C[SelectSheetWriteHandler]
    C --> D{@ExcelColumnSelect 注解}
    D -- 有 --> E[DictFrameworkUtils.getDictDataLabelList(...)]
    D -- 无 --> F[ExcelColumnSelectFunction.getOptions()]
    A --> G[ExcelUtils.read(...)]
    G --> H[DictConvert]
    H --> I[@DictFormat 注解]
    I --> J[DictFrameworkUtils.parseDictDataLabel(...)]
```

---

## 🧩 模块功能总结

| 包名            | 功能      | 关键类                         |
|---------------|---------|-----------------------------|
| `annotations` | 注解定义    | `DictFormat`                |
| `convert`     | 字段转换器   | `DictConvert`               |
| `function`    | 下拉数据源接口 | `ExcelColumnSelectFunction` |
| `handler`     | 写入处理器   | `SelectSheetWriteHandler`   |
| `util`        | 工具类封装   | `ExcelUtils`                |
| `dict.core`   | 字典转换工具  | `DictFrameworkUtils`        |

---

## 🧾 模块实现原理详解

### 1️⃣ 字典字段转换实现流程

- **步骤**：
    1. Controller 方法标注 `@DictFormat(SysDictTypeConstants.USER_STATUS)`
    2. 在 Excel 导出时，`DictConvert` 调用 `DictFrameworkUtils.parseDictDataLabel(...)` 将 `1` 转为 `"启用"`
    3. 在 Excel 导入时，`DictConvert` 调用 `DictFrameworkUtils.parseDictDataValue(...)` 将 `"启用"` 转为 `1`
    4. 使用 Guava 缓存避免重复请求远程服务

### 2️⃣ 下拉列生成实现流程

- **步骤**：
    1. DTO 字段添加 `@ExcelColumnSelect(dictType = "user.status")`
    2. 构造 `SelectSheetWriteHandler` 时解析字段
    3. 创建独立 Sheet 存储下拉数据
    4. 使用 Apache POI 设置 `DataValidationConstraint`
    5. Excel 导出后，用户只能选择预设值

### 3️⃣ 金额字段转换实现流程

- **步骤**：
    1. DTO 字段为 `Integer amount;`（单位：分）
    2. 添加 `@ExcelProperty` 和 `@Converter(MoneyConvert.class)`
    3. `MoneyConvert` 在导出时将 `100` 转为 `"1.00"`
    4. 导入时自动将 `"1.00"` 转回 `100`

---

## 🧪 单元测试与异常处理

### 示例：`DictFrameworkUtilsTest.java`

```java
@Test
public void testParseDictDataLabel() {
    List<DictDataRespDTO> dictDatas = List.of(
        randomPojo(DictDataRespDTO.class, o -> o.setDictType("animal").setValue("cat").setLabel("猫")),
        randomPojo(DictDataRespDTO.class, o -> o.setDictType("animal").setValue("dog").setLabel("狗"))
    );
    when(dictDataApi.getDictDataList(eq("animal"))).thenReturn(success(dictDatas));

    assertEquals("狗", DictFrameworkUtils.parseDictDataLabel("animal", "dog"));
}
```

- **作用**：验证字典转换器的准确性。
- **覆盖范围**：
    - 正常情况（含多字典类型）
    - 异常情况（如标签不存在、字典未加载）
- **测试覆盖率建议**：80%+

---

## ✅ 廁议改进方向

| 改进点      | 描述                                 |
|----------|------------------------------------|
| ✅ 多语言支持  | 当前仅支持中文，未来可扩展英文、日文等                |
| ✅ 性能优化   | 增加批量转换优化（减少单个字段转换开销）               |
| ✅ 单元测试增强 | 补充更多边界条件测试（如空值、非法输入）               |
| ✅ 分布式缓存  | 使用 Redis 替代 Guava Cache，避免集群不一致    |
| ✅ 多租户增强  | 结合 TenantContextHolder 实现租户级别的字典隔离 |

---

## 📌 总结

`pei-spring-boot-starter-excel` 模块实现了以下核心功能：

| 功能        | 技术实现                             | 用途                  |
|-----------|----------------------------------|---------------------|
| 字典字段转换    | DictConvert + DictFrameworkUtils | 自动将字典值转为可读标签        |
| 下拉列生成     | SelectSheetWriteHandler          | 提供下拉选择框，提升用户体验      |
| 金额字段转换    | MoneyConvert                     | 支持金额单位自动转换（分 → 元）   |
| 地区字段转换    | AreaConvert                      | 地区名称与编号自动互转         |
| JSON 字段转换 | JsonConvert                      | 支持任意对象的 JSON 显示     |
| 字典缓存      | DictFrameworkUtils + Guava       | 提升 Excel 转换性能       |
| 工具类封装     | ExcelUtils                       | 提供统一的 Excel 导入/导出入口 |

它是一个轻量但功能完整的 Excel 模块，适用于电商订单导出、会员信息管理、CRM 客户报表、ERP 物料清单等场景。

如果你有具体某个类（如 `DictConvert`、`SelectSheetWriteHandler`）想要深入了解，欢迎继续提问！
