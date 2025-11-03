# 第1章：Kotlin基础语法入门

## 📖 章节概述

本章将帮助已经精通Java的开发人员快速掌握Kotlin的基础语法。我们将从Java开发者的视角出发，重点讲解Kotlin与Java的差异，突出Kotlin的简洁性、安全性和表达能力。

**学习时长**: 约2-3天
**核心目标**: 熟练掌握Kotlin基础语法，能够编写简单的Kotlin程序

---

## 1.1 环境搭建与开发工具配置

### 1.1.1 开发环境要求

作为Java开发者，你已经具备了大部分Kotlin开发所需的基础环境：

- **JDK**: 17+（推荐OpenJDK 17）
- **IDE**: IntelliJ IDEA 2023.3+ 或 Android Studio
- **构建工具**: Gradle 8.0+

### 1.1.2 IntelliJ IDEA配置

IntelliJ IDEA对Kotlin提供了开箱即用的支持：

```kotlin
// build.gradle.kts
plugins {
    kotlin("jvm") version "2.0.20"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
}
```

### 1.1.3 Kotlin REPL快速体验

IntelliJ IDEA内置了Kotlin REPL，非常适合快速测试语法：

```kotlin
val message = "Hello, Kotlin!"
println(message)

// 立即看到输出结果
// Hello, Kotlin!
```

### 1.1.4 Kotlin Playground

对于不想配置环境的开发者，可以使用官方在线编辑器：
- 访问 [play.kotlinlang.org](https://play.kotlinlang.org/)
- 即时运行Kotlin代码
- 支持JavaScript和JVM目标平台

---

## 1.2 变量声明与类型推断

### 1.2.1 基本声明语法

Kotlin引入了两种变量声明关键字：

#### val - 不可变引用（相当于Java的final）

```kotlin
// Kotlin
val name: String = "张三"
val age = 25  // 类型推断，自动推断为Int

// 等价的Java代码
final String name = "张三";
final int age = 25;
```

#### var - 可变引用

```kotlin
// Kotlin
var score: Int = 100
score = 95  // 可以重新赋值

// 等价的Java代码
int score = 100;
score = 95;
```

### 1.2.2 类型推断详解

Kotlin的智能类型推断是相比Java的一大优势：

```kotlin
// 自动推断类型
val message = "Hello World"           // 推断为 String
val number = 42                       // 推断为 Int
val decimal = 3.14                    // 推断为 Double
val flag = true                       // 推断为 Boolean

// 复杂类型推断
val numbers = listOf(1, 2, 3, 4)      // 推断为 List<Int>
val userInfo = mapOf("name" to "李四", "age" to 30)  // 推断为 Map<String, Any>
```

### 1.2.3 最佳实践建议

```kotlin
// ✅ 推荐做法：尽可能使用val
val configuration = loadConfig()      // 不可变的配置
val database = connectToDatabase()    // 数据库连接引用

// ⚠️ 谨慎使用var：只在确实需要修改时使用
var counter = 0                      // 计数器需要修改
counter++                            // 这是合理的var使用场景

// ❌ 避免的做法：滥用var
var userList = loadUsers()           // 如果只是重新赋值，应该使用val
userList = processUsers(userList)    // 创建新列表而不是修改原列表
```

---

## 1.3 基本数据类型与Java对比

### 1.3.1 Kotlin的数据类型体系

Kotlin将所有类型都视为对象，没有Java中的基本类型与包装类型之分：

```kotlin
// Kotlin - 所有类型都是对象
val intNum: Int = 42          // Int类
val doubleNum: Double = 3.14  // Double类
val charValue: Char = 'A'     // Char类
val boolValue: Boolean = true // Boolean类

// Java对比
int intNum = 42;              // 基本类型
Integer intObj = 42;          // 包装类型
```

### 1.3.2 数据类型对应关系

| Kotlin类型 | Java类型 | 大小 | 说明 |
|-----------|----------|------|------|
| `Byte` | `byte` | 8位 | -128到127 |
| `Short` | `short` | 16位 | -32768到32767 |
| `Int` | `int` | 32位 | -2³¹到2³¹-1 |
| `Long` | `long` | 64位 | -2⁶³到2⁶³-1 |
| `Float` | `float` | 32位 | IEEE 754单精度 |
| `Double` | `double` | 64位 | IEEE 754双精度 |
| `Char` | `char` | 16位 | Unicode字符 |
| `Boolean` | `boolean` | - | true或false |

### 1.3.3 类型转换

Kotlin的类型转换比Java更明确和安全：

```kotlin
// Kotlin - 显式转换
val intVal: Int = 42
val longVal: Long = intVal.toLong()
val doubleVal: Double = intVal.toDouble()
val stringVal: String = intVal.toString()

// 自动类型安全的类型转换
val any: Any = "Hello"
if (any is String) {
    val length = any.length  // 智能转换，无需手动转换
}

// Java对比 - 自动类型提升（可能导致精度丢失）
int intVal = 42;
long longVal = intVal;        // 自动转换
double doubleVal = intVal;    // 自动转换
```

### 1.3.4 数字的表示方式

```kotlin
// Kotlin支持多种数字表示方式
val decimal = 123              // 十进制
val hex = 0x1F                 // 十六进制
val binary = 0b101010          // 二进制

// 类型后缀
val longNumber = 42L           // Long类型
val floatNumber = 3.14f        // Float类型
val doubleNumber = 3.14        // Double类型（默认）

// 使用下划线提高可读性（Kotlin 1.1+）
val million = 1_000_000        // 1,000,000
val creditCard = 1234_5678_9012_3456L
val bytes = 0b11010010_01101001_11010010_01101001
```

---

## 1.4 控制流语句的Kotlin式改进

### 1.4.1 if表达式

在Kotlin中，if不仅仅是语句，更是表达式：

```kotlin
// Kotlin - if作为表达式
val max = if (a > b) a else b

val message = if (score >= 90) {
    "优秀"
} else if (score >= 80) {
    "良好"
} else if (score >= 60) {
    "及格"
} else {
    "不及格"
}

// Java对比 - 需要使用三元运算符或if语句
int max = (a > b) ? a : b;

String message;
if (score >= 90) {
    message = "优秀";
} else if (score >= 80) {
    message = "良好";
} else if (score >= 60) {
    message = "及格";
} else {
    message = "不及格";
}
```

### 1.4.2 when表达式

when表达式是Java switch语句的升级版：

```kotlin
// Kotlin - when表达式
fun describe(obj: Any): String = when (obj) {
    1 -> "One"
    "Hello" -> "Greeting"
    is Long -> "Long number"
    !is String -> "Not a string"
    else -> "Unknown"
}

// 多条件匹配
val grade = when (score) {
    in 90..100 -> "A"
    in 80..89 -> "B"
    in 70..79 -> "C"
    in 60..69 -> "D"
    else -> "F"
}

// Java对比 - switch语句限制较多
switch (obj) {
    case 1:
        return "One";
    case "Hello":
        return "Greeting";
    default:
        return "Unknown";
}
```

### 1.4.3 循环语句

Kotlin简化了循环语法：

```kotlin
// for循环 - 更简洁的范围迭代
for (i in 1..10) {
    println(i)
}

// 步长控制
for (i in 1..10 step 2) {
    println(i)  // 输出1, 3, 5, 7, 9
}

// 降序迭代
for (i in 10 downTo 1) {
    println(i)
}

// 排除末尾
for (i in 1 until 10) {
    println(i)  // 输出1到9
}

// 集合迭代
val fruits = listOf("Apple", "Banana", "Orange")
for (fruit in fruits) {
    println(fruit)
}

// 带索引的集合迭代
for ((index, fruit) in fruits.withIndex()) {
    println("$index: $fruit")
}

// Java对比 - 传统的for循环
for (int i = 1; i <= 10; i++) {
    System.out.println(i);
}

String[] fruits = {"Apple", "Banana", "Orange"};
for (int i = 0; i < fruits.length; i++) {
    System.out.println(i + ": " + fruits[i]);
}
```

### 1.4.4 while和do-while循环

```kotlin
// 基本用法与Java相同
var x = 5
while (x > 0) {
    println(x)
    x--
}

do {
    println("至少执行一次")
} while (false)
```

---

## 1.5 空安全机制详解

空指针异常（NullPointerException）是Java中最常见的运行时异常之一。Kotlin通过类型系统从根本上解决了这个问题。

### 1.5.1 可空类型与非空类型

```kotlin
// Kotlin - 明确区分可空和非空类型
val nonNull: String = "Hello"      // 非空类型，不能赋值为null
val nullable: String? = null       // 可空类型，可以赋值为null

// Java - 所有引用类型都可为null
String nonNull = "Hello";          // 可能为null，编译器无法检查
String nullable = null;            // 可以为null
```

### 1.5.2 安全调用操作符 ?.

安全调用操作符是Kotlin空安全的明星特性：

```kotlin
// Kotlin - 安全调用
val name: String? = "张三"
val length = name?.length  // 如果name为null，则length为null，否则返回length

// 链式安全调用
val cityLength = user?.address?.city?.length

// Java对比 - 需要繁琐的null检查
String name = "张三";
Integer length = (name != null) ? name.length() : null;

String cityLength = null;
if (user != null && user.getAddress() != null &&
    user.getAddress().getCity() != null) {
    cityLength = user.getAddress().getCity().length();
}
```

### 1.5.3 Elvis操作符 ?:

Elvis操作符为空值提供默认值：

```kotlin
// Kotlin - Elvis操作符
val name: String? = null
val displayName = name ?: "Unknown"  // 如果name为null，使用"Unknown"

val length = name?.length ?: 0       // 如果name为null，length为0

// Java对比 - 需要三元运算符
String name = null;
String displayName = (name != null) ? name : "Unknown";

int length = (name != null) ? name.length() : 0;
```

### 1.5.4 非空断言操作符 !!

当你确定变量不为null时，可以使用非空断言：

```kotlin
// Kotlin - 非空断言（谨慎使用）
val name: String? = "张三"
val upperName = name!!.toUpperCase()  // 如果name为null，抛出NPE

// ❌ 危险用法：可能导致NPE
val risky: String? = null
val result = risky!!.length          // 抛出NPE

// 💡 推荐用法：结合let函数
val result = risky?.let { it.length } ?: 0
```

### 1.5.5 安全的类型转换

```kotlin
// Kotlin - 安全的类型转换
val obj: Any = "Hello"
val str: String? = obj as? String    // 安全转换，失败返回null

// Java对比 - 强制类型转换可能抛出ClassCastException
Object obj = "Hello";
String str = (String) obj;  // 可能抛出ClassCastException
```

### 1.5.6 空安全的实践建议

```kotlin
// ✅ 推荐做法：设计API时明确空值意图
fun findUser(id: Int): User? {
    // 明确返回可空类型，调用者知道需要处理null情况
    return database.findUser(id)
}

// ✅ 推荐做法：使用安全调用和Elvis操作符
val userName = user?.name ?: "Guest"

// ✅ 推荐做法：使用let函数处理可空值
email?.let { sendEmail(it) }

// ❌ 避免做法：滥用非空断言
val user = getUser()!!  // 危险，可能抛出NPE
```

---

## 1.6 字符串模板与多行字符串

### 1.6.1 字符串模板

Kotlin的字符串模板让字符串拼接变得异常简洁：

```kotlin
// Kotlin - 字符串模板
val name = "张三"
val age = 25
val message = "我叫$name，今年$age岁"

// 表达式模板
val score = 95
val result = "考试成绩：$score，${if (score >= 60) "及格" else "不及格"}"

// 复杂表达式
val price = 19.99
val quantity = 3
val total = "总价：${(price * quantity).format(2)}"

// Java对比 - 繁琐的字符串拼接
String name = "张三";
int age = 25;
String message = "我叫" + name + "，今年" + age + "岁";

// 或者使用String.format
String result = String.format("考试成绩：%d，%s",
    score, (score >= 60) ? "及格" : "不及格");
```

### 1.6.2 多行字符串

Kotlin原生支持多行字符串：

```kotlin
// Kotlin - 多行字符串
val json = """
    {
        "name": "张三",
        "age": 25,
        "email": "zhangsan@example.com"
    }
"""

// 去除缩进
val formattedJson = """
    {
        "name": "张三",
        "age": 25,
        "email": "zhangsan@example.com"
    }
""".trimIndent()

// 去除前导空格
val sql = """
    SELECT id, name, email
    FROM users
    WHERE age > ${minAge}
    ORDER BY name
    """.trimMargin()

// Java对比 - 需要使用+连接或StringBuilder
String json = "{\n" +
    "    \"name\": \"张三\",\n" +
    "    \"age\": 25,\n" +
    "    \"email\": \"zhangsan@example.com\"\n" +
    "}";
```

### 1.6.3 原始字符串与转义字符

```kotlin
// Kotlin - 原始字符串中的特殊字符处理
val path = "C:\\Users\\Admin\\Documents"  // 需要转义反斜杠
val regex = "\\d{3}-\\d{4}"              // 正则表达式
val quote = "他说：\"你好！\""            // 转义引号

// 或者使用原始字符串
val rawPath = """C:\Users\Admin\Documents"""
val rawRegex = """\d{3}-\d{4}"""
val rawQuote = """他说："你好！""""
```

---

## 1.7 Java vs Kotlin 语法对比速查表

| 特性 | Java | Kotlin | 改进点 |
|------|------|--------|--------|
| 变量声明 | `String name = "张三";` | `val name = "张三"` | 更简洁，支持类型推断 |
| 常量 | `final int MAX = 100;` | `const val MAX = 100` | 编译时常量支持 |
| 空值处理 | 需要手动null检查 | 内置空安全机制 | 编译时避免NPE |
| 字符串格式化 | `String.format("%s %d", name, age)` | `"$name $age"` | 更简洁直观 |
| 数据类 | 需要手写getter/setter等 | `data class User(val name: String)` | 自动生成样板代码 |
| switch语句 | `switch`仅支持基本类型 | `when`支持任意类型和条件 | 更强大灵活 |
| 函数定义 | `public String getName() { return name; }` | `fun getName(): String = name` | 表达式语法，更简洁 |
| 集合操作 | 需要外部库或手动实现 | 内置丰富的集合操作符 | 函数式编程支持 |

---

## 1.8 实战练习：Java到Kotlin的代码转换

### 练习1：简单业务逻辑转换

**Java原始代码**：
```java
public class UserService {
    private String userName;
    private Integer userAge;

    public UserService(String name, Integer age) {
        this.userName = name;
        this.userAge = age;
    }

    public String getDisplayName() {
        if (userName != null && !userName.trim().isEmpty()) {
            return userName;
        }
        return "Unknown User";
    }

    public String getAgeDescription() {
        if (userAge != null) {
            if (userAge < 18) {
                return "未成年";
            } else if (userAge < 60) {
                return "成年";
            } else {
                return "老年";
            }
        }
        return "年龄未知";
    }

    public boolean isValidUser() {
        return userName != null && !userName.trim().isEmpty()
            && userAge != null && userAge > 0;
    }
}
```

**Kotlin转换后的代码**：
```kotlin
class UserService(private val userName: String?, private val userAge: Int?) {
    fun getDisplayName(): String = userName?.takeIf { it.isNotBlank() } ?: "Unknown User"

    fun getAgeDescription(): String = when {
        userAge == null -> "年龄未知"
        userAge < 18 -> "未成年"
        userAge < 60 -> "成年"
        else -> "老年"
    }

    fun isValidUser(): Boolean =
        !userName.isNullOrBlank() && (userAge ?: 0) > 0
}
```

### 练习2：数据转换与处理

**Java版本**：
```java
public class DataProcessor {
    public List<String> processScores(List<Integer> scores) {
        List<String> results = new ArrayList<>();
        if (scores != null) {
            for (Integer score : scores) {
                if (score != null && score >= 60) {
                    String grade = getGrade(score);
                    results.add("分数: " + score + ", 等级: " + grade);
                }
            }
        }
        return results;
    }

    private String getGrade(int score) {
        if (score >= 90) return "优秀";
        if (score >= 80) return "良好";
        if (score >= 70) return "中等";
        return "及格";
    }
}
```

**Kotlin版本**：
```kotlin
class DataProcessor {
    fun processScores(scores: List<Int?>?): List<String> {
        return scores?.mapNotNull { score ->
            score?.takeIf { it >= 60 }?.let {
                "分数: $score, 等级: ${getGrade(it)}"
            }
        } ?: emptyList()
    }

    private fun getGrade(score: Int): String = when {
        score >= 90 -> "优秀"
        score >= 80 -> "良好"
        score >= 70 -> "中等"
        else -> "及格"
    }
}
```

### 练习3：配置文件处理

**Java版本**：
```java
public class ConfigManager {
    private Properties properties = new Properties();

    public void loadConfig(String configPath) {
        try {
            properties.load(new FileInputStream(configPath));
        } catch (IOException e) {
            System.err.println("配置文件加载失败: " + e.getMessage());
        }
    }

    public String getServerUrl() {
        String url = properties.getProperty("server.url");
        String port = properties.getProperty("server.port");
        String protocol = properties.getProperty("server.protocol");

        if (url == null || port == null || protocol == null) {
            return "http://localhost:8080";
        }

        return protocol + "://" + url + ":" + port;
    }

    public int getConnectionTimeout() {
        String timeout = properties.getProperty("connection.timeout");
        if (timeout != null && !timeout.trim().isEmpty()) {
            try {
                return Integer.parseInt(timeout);
            } catch (NumberFormatException e) {
                return 30000; // 默认30秒
            }
        }
        return 30000;
    }
}
```

**Kotlin版本**：
```kotlin
class ConfigManager {
    private val properties = Properties()

    fun loadConfig(configPath: String) {
        try {
            FileInputStream(configPath).use { input ->
                properties.load(input)
            }
        } catch (e: IOException) {
            println("配置文件加载失败: ${e.message}")
        }
    }

    fun getServerUrl(): String {
        val url = properties.getProperty("server.url")
        val port = properties.getProperty("server.port")
        val protocol = properties.getProperty("server.protocol")

        return if (listOf(url, port, protocol).all { it != null }) {
            "$protocol://$url:$port"
        } else {
            "http://localhost:8080"
        }
    }

    fun getConnectionTimeout(): Int {
        return properties.getProperty("connection.timeout")?.trim()?.let {
            it.toIntOrNull() ?: 30000
        } ?: 30000
    }
}
```

---

## 1.9 本章小结

通过本章的学习，您已经掌握了：

### ✅ 核心概念
- **变量声明**：val vs var，类型推断机制
- **数据类型**：统一的类型系统，显式类型转换
- **控制流**：if/when表达式，简化的循环语法
- **空安全**：编译时避免NullPointerException
- **字符串处理**：模板和多行字符串

### ✅ 关键优势
- **简洁性**：相比Java代码量减少40%以上
- **安全性**：编译时捕获空指针异常
- **表达力**：更直观的语法和操作符
- **互操作性**：与Java代码无缝集成

### ✅ 实践要点
- 优先使用val而非var
- 充分利用类型推断
- 使用空安全操作符而非非空断言
- 采用表达式语法替代语句
- 利用字符串模板简化字符串操作

### 📚 下一步学习
下一章我们将深入探讨Kotlin的面向对象编程特性，包括类、继承、接口、数据类等概念，以及它们相比Java的改进和优势。

---

## 📝 章节练习

### 基础练习
1. 将以下Java代码转换为Kotlin：
```java
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public boolean isPositive(Integer number) {
        return number != null && number > 0;
    }

    public String formatNumber(Integer number) {
        if (number == null) {
            return "N/A";
        }
        return "Number: " + number;
    }
}
```

2. 编写一个Kotlin函数，接收一个字符串列表，返回所有非空字符串的长度列表。

### 进阶练习
1. 实现一个简单的用户验证类，要求：
- 使用Kotlin的数据类
- 实现空安全的邮箱验证
- 使用when表达式进行年龄分类
- 使用字符串模板生成用户信息

2. 创建一个配置管理类，能够：
- 从Map中读取配置项
- 提供默认值处理
- 使用安全调用避免NPE
- 生成配置信息的多行字符串报告

---

**恭喜！您已经完成了Kotlin基础语法的学习。继续保持这个学习节奏，下一章我们将探索Kotlin面向对象编程的魅力！**