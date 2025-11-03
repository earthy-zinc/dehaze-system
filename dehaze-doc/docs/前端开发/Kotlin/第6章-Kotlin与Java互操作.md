# 第6章：Kotlin与Java互操作

## 📖 章节概述

在实际项目中，Kotlin很少作为完全独立的语言存在，更多的是与Java代码混合使用。本章将深入探讨Kotlin与Java之间的互操作机制，帮助您在现有Java项目中平滑引入Kotlin，或者在新项目中实现两者的完美协作。

**学习时长**: 约3-4天
**核心目标**: 掌握Kotlin与Java互操作的所有技巧，能够无缝地在两种语言间切换和集成

---

## 6.1 Kotlin调用Java代码

### 6.1.1 基础互操作

```kotlin
// Kotlin调用Java基础示例
fun basicJavaInteropDemo() {
    // Java类的使用
    val arrayList = ArrayList<String>()
    arrayList.add("Kotlin")
    arrayList.add("Java")
    arrayList.add("Interoperability")

    println("ArrayList内容: $arrayList")

    // Java静态方法调用
    val currentTime = System.currentTimeMillis()
    println("当前时间戳: $currentTime")

    // Java枚举使用
    val day = java.time.DayOfWeek.MONDAY
    println("星期枚举: $day")

    // Java接口实现
    val runnable = Runnable {
        println("Runnable executed in Kotlin")
    }
    runnable.run()

    // Java异常处理
    try {
        val url = URL("https://www.example.com")
        println("URL协议: ${url.protocol}")
    } catch (e: MalformedURLException) {
        println("URL格式错误: ${e.message}")
    }

    // Java集合的使用
    val javaMap = HashMap<String, Int>()
    javaMap["one"] = 1
    javaMap["two"] = 2
    javaMap["three"] = 3

    println("HashMap: $javaMap")

    // Java工具类使用
    val collections = Collections.singletonList("Single Element")
    println("单个元素集合: $collections")
}
```

### 6.1.2 处理Java的可空性

```kotlin
// Java可空性处理
fun javaNullabilityDemo() {
    // Java中的@Nullable注解
    // 假设有Java类：
    // public class JavaService {
    //     @Nullable
    //     public String getData() { return null; }
    // }

    val javaService = JavaService()

    // Kotlin知道Java方法可能返回null
    val data: String? = javaService.getData()
    println("Java服务数据: $data")

    // 安全调用
    val length = javaService.getData()?.length ?: 0
    println("数据长度: $length")

    // 平台类型处理
    // 当Java代码没有注解时，Kotlin使用平台类型
    val platformTypeString: String! = javaService.getUnannotatedString()
    println("平台类型字符串: $platformTypeString")

    // 安全处理平台类型
    val safeLength = platformTypeString?.length ?: 0
    println("安全长度: $safeLength")
}

// 模拟Java类用于演示
class JavaService {
    @Nullable
    fun getData(): String? = null

    fun getUnannotatedString(): String = "Hello from Java"
}

@Target(AnnotationTarget.VALUE_PARAMETER, AnnotationTarget.FIELD,
        AnnotationTarget.METHOD, AnnotationTarget.TYPE)
@Retention(AnnotationRetention.RUNTIME)
annotation class Nullable
```

### 6.1.3 Java集合与Kotlin集合的互操作

```kotlin
// Java和Kotlin集合互操作
fun collectionInteropDemo() {
    // Java集合到Kotlin集合
    val javaList = java.util.ArrayList<String>()
    javaList.add("Java")
    javaList.add("Kotlin")
    javaList.add("Scala")

    // 自动转换
    val kotlinList: List<String> = javaList
    println("Kotlin只读列表: $kotlinList")

    // 可变集合的转换
    val kotlinMutableList: MutableList<String> = javaList
    kotlinMutableList.add("Groovy")
    println("Kotlin可变列表: $kotlinMutableList")

    // Java Map的操作
    val javaMap = java.util.HashMap<String, Integer>()
    javaMap["one"] = 1
    javaMap["two"] = 2
    javaMap["three"] = 3

    val kotlinMap: Map<String, Int> = javaMap
    println("Kotlin Map: $kotlinMap")

    // 使用Kotlin扩展函数操作Java集合
    javaList.filter { it.startsWith("J") }
        .forEach { println("Java开头: $it") }

    javaMap.filterValues { it > 1 }
        .forEach { (key, value) -> println("$key: $value") }

    // 集合类型的注意事项
    fun processJavaCollection(collection: java.util.Collection<String>) {
        val kotlinList = collection.toList()
        val processed = kotlinList.map { it.uppercase() }
        println("处理后: $processed")
    }

    processJavaCollection(javaList)

    // 性能考虑：避免不必要的转换
    fun efficientProcessing(javaList: java.util.List<String>) {
        // 直接在Java集合上使用Kotlin扩展
        javaList.asSequence()
            .filter { it.length > 3 }
            .map { it.lowercase() }
            .forEach { println("高效处理: $it") }
    }

    efficientProcessing(javaList)
}
```

### 6.1.4 SAM转换

```kotlin
// SAM (Single Abstract Method) 转换
fun samConversionDemo() {
    // 传统方式创建匿名类
    val traditionalRunnable = object : Runnable {
        override fun run() {
            println("Traditional Runnable")
        }
    }

    // SAM转换 - 更简洁
    val samRunnable = Runnable {
        println("SAM Runnable")
    }

    traditionalRunnable.run()
    samRunnable.run()

    // Java 8函数式接口
    val comparator = java.util.Comparator<String> { a, b ->
        a.compareTo(b)
    }

    val strings = listOf("Zebra", "Apple", "Banana", "Cherry")
    val sortedStrings = strings.sortedWith(comparator)
    println("SAM排序: $sortedStrings")

    // 复杂SAM转换
    val consumer = java.util.function.Consumer<String> { item ->
        println("Consuming: $item")
    }

    strings.forEach(consumer)

    val predicate = java.util.function.Predicate<String> { str ->
        str.startsWith("A")
    }

    val filteredStrings = strings.filter(predicate)
    println("SAM过滤: $filteredStrings")

    // 自定义SAM接口
    val clickListener = View.OnClickListener { view ->
        println("View clicked: $view")
    }

    val view = View("Button")
    view.setOnClickListener(clickListener)
    view.performClick()

    // 多个抽象方法的情况不能使用SAM转换
    val multiMethodListener = object : View.OnLongClickListener, View.OnClickListener {
        override fun onLongClick(v: View?): Boolean {
            println("Long click")
            return true
        }

        override fun onClick(v: View?) {
            println("Click")
        }
    }

    view.setOnLongClickListener(multiMethodListener)
    view.setOnClickListener(multiMethodListener)
}

// 模拟Android View类
class View(val name: String) {
    interface OnClickListener {
        fun onClick(v: View)
    }

    interface OnLongClickListener {
        fun onLongClick(v: View?): Boolean
    }

    private var clickListener: OnClickListener? = null
    private var longClickListener: OnLongClickListener? = null

    fun setOnClickListener(listener: OnClickListener) {
        this.clickListener = listener
    }

    fun setOnLongClickListener(listener: OnLongClickListener) {
        this.longClickListener = listener
    }

    fun performClick() {
        clickListener?.onClick(this)
    }

    fun performLongClick(): Boolean {
        return longClickListener?.onLongClick(this) ?: false
    }

    override fun toString(): String = "View($name)"
}
```

---

## 6.2 Java调用Kotlin代码

### 6.2.1 Kotlin代码的Java可见性

```kotlin
// 为Java调用优化的Kotlin类
// File: KotlinCalculator.kt

class KotlinCalculator {
    // @JvmField - 将属性暴露为Java字段
    @JvmField
    val version = "1.0.0"

    // 普通属性 - 在Java中通过getter访问
    val lastResult: Double
        get() = _lastResult

    private var _lastResult = 0.0

    // @JvmOverloads - 为Java生成重载方法
    @JvmOverloads
    fun add(a: Double, b: Double = 0.0): Double {
        _lastResult = a + b
        return _lastResult
    }

    // @JvmStatic - 生成静态方法
    companion object {
        @JvmStatic
        fun createInstance(): KotlinCalculator {
            return KotlinCalculator()
        }

        @JvmField
        val PI = 3.14159265359

        // 这个方法没有@JvmStatic，在Java中需要通过Companion访问
        fun getCalculatorInfo(): String {
            return "Kotlin Calculator v1.0"
        }
    }

    // 顶层函数 - 在Java中通过文件名调用
    fun calculateTax(amount: Double, rate: Double): Double {
        return amount * rate
    }
}

// 顶层函数
fun calculateDiscount(price: Double, discountPercent: Double): Double {
    return price * (1 - discountPercent / 100)
}

// object类
object MathUtils {
    @JvmStatic
    fun square(n: Double): Double = n * n

    @JvmStatic
    fun cube(n: Double): Double = n * n * n
}

// data class
@JvmName("createUserInfo")
data class UserInfo(
    val id: String,
    val name: String,
    var age: Int,
    @JvmField var email: String // 在Java中直接访问
)
```

### 6.2.2 Java调用Kotlin的示例

```java
// Java调用Kotlin代码的示例
// File: JavaInteropDemo.java

import java.util.*;

public class JavaInteropDemo {
    public static void main(String[] args) {
        // 调用KotlinCalculator
        KotlinCalculator calculator = new KotlinCalculator();

        // 访问@JvmField属性
        System.out.println("Calculator version: " + calculator.version);

        // 访问普通属性（通过getter）
        System.out.println("Initial last result: " + calculator.getLastResult());

        // 调用@JvmOverloads方法
        double result1 = calculator.add(5.0); // 使用默认参数
        double result2 = calculator.add(3.0, 4.0); // 提供所有参数

        System.out.println("5 + 0 = " + result1);
        System.out.println("3 + 4 = " + result2);
        System.out.println("Last result: " + calculator.getLastResult());

        // 调用companion object中的@JvmStatic方法
        KotlinCalculator staticCalculator = KotlinCalculator.createInstance();
        System.out.println("PI value: " + KotlinCalculator.PI);

        // 调用没有@JvmStatic的方法
        String info = KotlinCalculator.Companion.getCalculatorInfo();
        System.out.println("Calculator info: " + info);

        // 调用顶层函数
        double discountedPrice = KotlinInteropDemoKt.calculateDiscount(100.0, 20.0);
        System.out.println("Discounted price: " + discountedPrice);

        // 调用object中的方法
        double squared = MathUtils.square(5.0);
        double cubed = MathUtils.cube(3.0);
        System.out.println("5 squared: " + squared);
        System.out.println("3 cubed: " + cubed);

        // 使用data class
        UserInfo user = UserInfoKt.createUserInfo("001", "Alice", 25, "alice@example.com");
        System.out.println("User: " + user.getName() + ", " + user.getAge());

        // 直接访问@JvmField属性
        System.out.println("User email: " + user.email);
        user.email = "alice.new@example.com";
        System.out.println("Updated email: " + user.email);

        // 在集合中使用Kotlin对象
        List<UserInfo> users = new ArrayList<>();
        users.add(user);
        users.add(UserInfoKt.createUserInfo("002", "Bob", 30, "bob@example.com"));

        System.out.println("Users count: " + users.size());

        // 处理异常
        try {
            // 假设Kotlin方法可能抛出异常
            calculator.add(Double.MAX_VALUE, Double.MAX_VALUE);
        } catch (Exception e) {
            System.out.println("Exception from Kotlin: " + e.getMessage());
        }
    }
}
```

### 6.2.3 注解处理和元数据

```kotlin
// Kotlin中的Java注解使用
import java.lang.annotation.*

// 自定义注解
@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
annotation class ApiEndpoint(
    val path: String,
    val method: String = "GET"
)

// 使用Java注解
@Deprecated("Use NewUserService instead")
@ApiEndpoint("/users", method = "GET")
@Retention(RetentionPolicy.RUNTIME)
class UserService {
    @JvmName("findUserById")
    fun getUser(@JvmName("userId") id: String): User? {
        return User(id, "User $id")
    }
}

// 为Java友好的注解使用
data class User(
    @field:SerializedName("user_id") // Jackson/Gson注解
    val id: String,

    @field:SerializedName("user_name")
    val name: String
)

// 处理Java泛型
class Repository {
    // 使用通配符为Java提供更好的类型安全性
    @JvmSuppressWildcards
    fun getUsers(): List<User> {
        return listOf(User("1", "Alice"), User("2", "Bob"))
    }

    // 为Java提供具体类型
    @JvmName("getUserMap")
    fun getUserMap(): Map<String, User> {
        return getUsers().associateBy { it.id }
    }
}
```

---

## 6.3 注解处理兼容性

### 6.3.1 KAPT vs KSP

```kotlin
// 注解处理器配置对比

// build.gradle.kts - KAPT配置
plugins {
    kotlin("kapt")
}

dependencies {
    kapt("com.google.auto.service:auto-service:1.0.1")
    implementation("com.google.auto.service:auto-service-annotations:1.0.1")
}

// build.gradle.kts - KSP配置
plugins {
    id("com.google.devtools.ksp") version "1.9.0-1.0.13"
}

dependencies {
    ksp("com.google.auto.service:auto-service:1.0.1")
    implementation("com.google.auto.service:auto-service-annotations:1.0.1")
}

// 使用注解的Kotlin代码
@AutoService(Processor::class)
class MyAnnotationProcessor : AbstractProcessor() {
    override fun process(annotations: Set<TypeElement>, roundEnv: RoundEnvironment): Boolean {
        // 处理注解逻辑
        return true
    }

    override fun getSupportedAnnotationTypes(): Set<String> {
        return setOf("com.example.MyAnnotation")
    }
}

// 自定义注解
@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.SOURCE)
annotation class MyAnnotation(
    val value: String,
    val version: Int = 1
)

// 使用自定义注解
@MyAnnotation("Generated class", version = 2)
class GeneratedClass
```

### 6.3.2 Lombok集成

```kotlin
// Kotlin中使用Lombok生成的Java类

// 假设有Java类使用Lombok：
/*
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class JavaPerson {
    private String name;
    private Integer age;
    private String email;
}
*/

// 在Kotlin中使用Lombok生成的Java类
fun lombokInteropDemo() {
    // 使用builder模式
    val person = JavaPerson.builder()
        .name("Alice")
        .age(25)
        .email("alice@example.com")
        .build()

    // 访问属性（通过getter/setter）
    println("Name: ${person.name}")
    println("Age: ${person.age}")

    // 修改属性
    person.age = 26
    println("Updated age: ${person.age}")

    // toString方法（Lombok自动生成）
    println("Person info: $person")
}

// Kotlin的data class可以替代Lombok
data class KotlinPerson(
    val name: String,
    var age: Int,
    val email: String
)

// 为与Lombok互操作而设计的Kotlin类
class CompatiblePerson {
    @JvmField var name: String
    @JvmField var age: Int
    @JvmField var email: String

    constructor(name: String, age: Int, email: String) {
        this.name = name
        this.age = age
        this.email = email
    }

    // 为Java提供getter/setter
    fun getName(): String = name
    fun setName(name: String) { this.name = name }
    fun getAge(): Int = age
    fun setAge(age: Int) { this.age = age }
    fun getEmail(): String = email
    fun setEmail(email: String) { this.email = email }

    // 重写equals、hashCode、toString
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is CompatiblePerson) return false
        return name == other.name && age == other.age && email == other.email
    }

    override fun hashCode(): Int {
        return Objects.hash(name, age, email)
    }

    override fun toString(): String {
        return "CompatiblePerson(name=$name, age=$age, email=$email)"
    }
}
```

---

## 6.4 SAM转换与函数式接口

### 6.4.1 Java函数式接口的Kotlin实现

```kotlin
// Kotlin实现Java函数式接口
fun functionalInterfaceDemo() {
    // Runnable接口
    val kotlinRunnable = Runnable {
        println("Kotlin implementation of Runnable")
    }

    // Consumer接口
    val kotlinConsumer = java.util.function.Consumer<String> { str ->
        println("Consuming: $str")
    }

    // Supplier接口
    val kotlinSupplier = java.util.function.Supplier<String> {
        "Hello from Supplier"
    }

    // Function接口
    val kotlinFunction = java.util.function.Function<String, Int> { str ->
        str.length
    }

    // Predicate接口
    val kotlinPredicate = java.util.function.Predicate<String> { str ->
        str.isNotBlank()
    }

    // 使用这些函数式接口
    executeRunnable(kotlinRunnable)
    consumeString(kotlinConsumer, "Hello World")
    println(supplyString(kotlinSupplier))
    println("String length: ${applyFunction(kotlinFunction, "Kotlin")}")
    println("String is not blank: ${testPredicate(kotlinPredicate, "Test")}")

    // 复杂的函数式接口组合
    val combinedPredicate = kotlinPredicate.and { str -> str.length > 3 }
    println("Combined predicate result: ${testPredicate(combinedPredicate, "Kotlin")}")

    // Optional的使用
    val optional: java.util.Optional<String> = java.util.Optional.of("Kotlin")
    optional
        .map(String::uppercase)
        .filter { it.startsWith("K") }
        .ifPresentOrElse(
            { println("Filtered result: $it") },
            { println("No result") }
        )
}

// 辅助函数
fun executeRunnable(runnable: Runnable) {
    runnable.run()
}

fun consumeString(consumer: java.util.function.Consumer<String>, value: String) {
    consumer.accept(value)
}

fun supplyString(supplier: java.util.function.Supplier<String>): String {
    return supplier.get()
}

fun applyFunction(function: java.util.function.Function<String, Int>, value: String): Int {
    return function.apply(value)
}

fun testPredicate(predicate: java.util.function.Predicate<String>, value: String): Boolean {
    return predicate.test(value)
}
```

### 6.4.2 自定义函数式接口

```kotlin
// 自定义函数式接口
fun customFunctionalInterfaceDemo() {
    // 定义和使用自定义函数式接口
    val calculator = Calculator { a, b -> a + b }
    val result = calculator.calculate(3, 4)
    println("Calculator result: $result")

    // 带泛型的函数式接口
    val transformer = Transformer<String, Int> { str -> str.length }
    val length = transformer.transform("Kotlin")
    println("String length: $length")

    // 多参数函数式接口
    val validator = Validator { name, age, email ->
        name.isNotBlank() && age > 0 && email.contains("@")
    }

    val isValid = validator.validate("Alice", 25, "alice@example.com")
    println("Validation result: $isValid")
}

// 自定义函数式接口定义
@FunctionalInterface
interface Calculator {
    fun calculate(a: Int, b: Int): Int
}

@FunctionalInterface
interface Transformer<T, R> {
    fun transform(input: T): R
}

@FunctionalInterface
interface Validator {
    fun validate(name: String, age: Int, email: String): Boolean
}

// 在Java中使用的Kotlin函数式接口
@FunctionalInterface
interface ClickListener {
    fun onClick(view: View)
    fun onLongClick(view: View): Boolean = false // 默认方法
}

// 在Kotlin中使用这个接口
fun clickListenerDemo() {
    val clickListener = ClickListener { view ->
        println("Clicked on: $view")
    }

    clickListener.onClick(View("Button"))

    // 调用默认方法
    val longClickResult = clickListener.onLongClick(View("Button"))
    println("Long click result: $longClickResult")
}
```

---

## 6.5 集合互操作

### 6.5.1 Java集合与Kotlin集合的转换

```kotlin
// 集合互操作详解
fun collectionInteropDemo() {
    // Java集合到Kotlin集合
    val javaList = java.util.ArrayList<String>()
    javaList.add("Java")
    javaList.add("Kotlin")
    javaList.add("Scala")
    javaList.add("Groovy")

    // 自动转换
    val kotlinReadOnlyList: List<String> = javaList
    val kotlinMutableList: MutableList<String> = javaList

    println("只读列表: $kotlinReadOnlyList")
    println("可变列表: $kotlinMutableList")

    // Java Map到Kotlin Map
    val javaMap = java.util.HashMap<String, Integer>()
    javaMap["one"] = 1
    javaMap["two"] = 2
    javaMap["three"] = 3

    val kotlinMap: Map<String, Int> = javaMap
    println("Map转换: $kotlinMap")

    // 使用Kotlin扩展函数操作Java集合
    val filteredJavaList = javaList.filter { it.startsWith("J") }
    println("过滤Java开头的: $filteredJavaList")

    val transformedMap = javaMap.mapValues { it.value.toString() }
    println("转换后的Map: $transformedMap")

    // 性能考虑
    val largeJavaList = (1..100000).toCollection(java.util.ArrayList())

    // 避免：重复转换
    val expensiveList = largeJavaList.toList() // 创建新集合
    val result1 = expensiveList.filter { it % 2 == 0 }
    val result2 = result1.map { it * it }

    // 推荐：直接在Java集合上操作
    val efficientResult = largeJavaList.asSequence()
        .filter { it % 2 == 0 }
        .map { it * it }
        .take(10)
        .toList()

    println("高效处理结果: $efficientResult")
}

// 集合类型的注意事项
fun collectionTypeConsiderations() {
    // 可变性的处理
    fun processJavaCollection(collection: java.util.Collection<String>) {
        // 这会创建一个快照，不会修改原集合
        val kotlinList = collection.toList()
        val processed = kotlinList.map { it.uppercase() }

        // 如果需要修改原集合，需要转换回去
        if (collection is java.util.MutableList<String>) {
            collection.clear()
            collection.addAll(processed)
        }
    }

    val mutableCollection = java.util.ArrayList(listOf("a", "b", "c"))
    processJavaCollection(mutableCollection)
    println("处理后的集合: $mutableCollection")

    // 类型安全考虑
    fun safeCollectionProcessing(javaList: java.util.List<*>) {
        // 使用星号投影处理泛型擦除
        val stringList = javaList.filterIsInstance<String>()
        stringList.forEach { println("String: $it") }
    }

    val mixedList = java.util.ArrayList<Any>()
    mixedList.add("Hello")
    mixedList.add(123)
    mixedList.add(true)

    safeCollectionProcessing(mixedList)
}
```

### 6.5.2 集合操作的桥接

```kotlin
// 集合操作桥接
fun collectionBridgeDemo() {
    // Java Stream与Kotlin Collection的桥接
    val javaList = java.util.Arrays.asList("apple", "banana", "cherry", "date")

    // Java Stream转换为Kotlin Collection
    val kotlinList = javaList.stream()
        .filter { it.startsWith("a") }
        .collect(java.util.stream.Collectors.toList())

    println("Stream过滤结果: $kotlinList")

    // Kotlin Collection转换为Java Stream
    val kotlinList2 = listOf("element1", "element2", "element3")
    val javaStream = kotlinList2.stream()

    val javaResult = javaStream
        .map { it.uppercase() }
        .collect(java.util.stream.Collectors.toList())

    println("Kotlin to Java Stream: $javaResult")

    // 自定义桥接函数
    fun <T> javaStreamToKotlinList(stream: java.util.stream.Stream<T>): List<T> {
        return stream.collect(java.util.stream.Collectors.toList())
    }

    fun <T> kotlinListToJavaStream(list: List<T>): java.util.stream.Stream<T> {
        return list.stream()
    }

    // 使用桥接函数
    val originalList = listOf("x", "y", "z")
    val stream = kotlinListToJavaStream(originalList)
    val processed = javaStreamToKotlinList(stream.map { "$item-processed" })

    println("桥接处理结果: $processed")

    // 集合工厂方法的互操作
    val javaCollection = java.util.Collections.singleton("single")
    val kotlinListFromJava = javaCollection.toList()
    println("Java单例集合: $kotlinListFromJava")

    val kotlinSet = setOf("a", "b", "c")
    val javaSetFromKotlin = java.util.HashSet(kotlinSet)
    println("Kotlin到Java Set: $javaSetFromKotlin")
}
```

---

## 6.6 异常处理的差异

### 6.1.1 Java检查异常在Kotlin中的处理

```kotlin
// Java检查异常的处理
fun checkedExceptionHandling() {
    // Java代码可能有检查异常
    /*
    public class JavaFileReader {
        public String readFile(String path) throws IOException {
            return Files.readString(Paths.get(path));
        }
    }
    */

    val fileReader = JavaFileReader()

    // 方法1：使用try-catch
    try {
        val content = fileReader.readFile("test.txt")
        println("File content: $content")
    } catch (e: IOException) {
        println("Error reading file: ${e.message}")
    }

    // 方法2：使用扩展函数包装
    fun JavaFileReader.readFileSafe(path: String): String? {
        return try {
            readFile(path)
        } catch (e: IOException) {
            println("Warning: ${e.message}")
            null
        }
    }

    val safeContent = fileReader.readFileSafe("test.txt")
    println("Safe content: $safeContent")

    // 方法3：使用Result类型
    fun JavaFileReader.readFileResult(path: String): Result<String> {
        return try {
            Result.success(readFile(path))
        } catch (e: IOException) {
            Result.failure(e)
        }
    }

    val result = fileReader.readFileResult("test.txt")
    result.onSuccess { content -> println("Success: $content") }
        .onFailure { error -> println("Failed: ${error.message}") }
}

// 模拟JavaFileReader
class JavaFileReader {
    @Throws(IOException::class)
    fun readFile(path: String): String {
        // 模拟可能的文件读取异常
        if (path == "error.txt") {
            throw IOException("File not found: $path")
        }
        return "Content of $path"
    }
}
```

### 6.1.2 异常传播机制

```kotlin
// 异常传播机制对比
fun exceptionPropagationDemo() {
    // Kotlin的异常传播
    suspend fun suspendFunctionWithException(): String {
        delay(100)
        throw IllegalStateException("Suspend function error")
    }

    // 在协程中处理异常
    try {
        runBlocking {
            suspendFunctionWithException()
        }
    } catch (e: IllegalStateException) {
        println("Caught in Kotlin: ${e.message}")
    }

    // Java到Kotlin的异常传播
    fun callJavaMethod(): String {
        val javaService = JavaService()
        return try {
            javaService.methodThatThrowsException()
        } catch (e: JavaCheckedException) {
            "Handled: ${e.message}"
        }
    }

    val result = callJavaMethod()
    println("Java exception handling result: $result")

    // 异常链的处理
    fun wrapException(action: () -> Unit) {
        try {
            action()
        } catch (e: Exception) {
            throw RuntimeException("Wrapped exception", e)
        }
    }

    try {
        wrapException {
            val javaService = JavaService()
            javaService.methodThatThrowsException()
        }
    } catch (e: RuntimeException) {
        println("Wrapped exception: ${e.message}")
        println("Original cause: ${e.cause?.message}")
    }
}

// 模拟Java服务类
class JavaService {
    @Throws(JavaCheckedException::class)
    fun methodThatThrowsException(): String {
        throw JavaCheckedException("Java checked exception occurred")
    }
}

class JavaCheckedException(message: String) : Exception(message)
```

---

## 6.7 实战项目：Java项目Kotlin化

### 6.7.1 渐进式迁移策略

```kotlin
// 渐进式迁移策略示例

// 1. 工具类迁移 - 首先迁移无状态的工具类
object StringUtils {
    @JvmStatic
    fun isBlank(str: String?): Boolean = str.isNullOrBlank()

    @JvmStatic
    fun capitalize(str: String): String = str.replaceFirstChar {
        if (it.isLowerCase()) it.uppercase() else it.toString()
    }

    @JvmStatic
    fun truncate(str: String, maxLength: Int): String {
        return if (str.length <= maxLength) str
        else str.take(maxLength - 3) + "..."
    }
}

// 2. 数据模型迁移 - 使用data class替代POJO
// 原Java POJO:
/*
public class User {
    private String id;
    private String name;
    private int age;

    // getters and setters...
    // equals, hashCode, toString...
}
*/

// Kotlin data class版本
@JvmName("createUser")
data class User(
    @JvmField val id: String,  // @JvmField为Java提供直接访问
    val name: String,
    var age: Int
) {
    // 为Java提供的额外方法
    fun isAdult(): Boolean = age >= 18

    fun getDisplayName(): String = "$name ($age岁)"

    companion object {
        @JvmStatic
        fun fromJson(json: String): User {
            // JSON解析逻辑
            return User("001", "Default", 0)
        }
    }
}

// 3. 服务层迁移 - 保持接口兼容性
interface UserService {
    fun getUser(id: String): User?
    fun saveUser(user: User): Boolean
    fun getAllUsers(): List<User>
}

// Kotlin实现
class KotlinUserService : UserService {
    private val users = mutableMapOf<String, User>()

    override fun getUser(id: String): User? = users[id]

    override fun saveUser(user: User): Boolean {
        users[user.id] = user
        return true
    }

    override fun getAllUsers(): List<User> = users.values.toList()

    // Kotlin特有的便利方法
    fun findAdults(): List<User> = users.values.filter { it.isAdult() }
    fun countByAgeGroup(): Map<String, Int> {
        return users.values.groupBy {
            when {
                it.age < 18 -> "Minor"
                it.age < 65 -> "Adult"
                else -> "Senior"
            }
        }.mapValues { it.value.size }
    }
}

// 4. 控制器层迁移 - 保持REST API兼容性
@RestController
@RequestMapping("/api/users")
class UserController(private val userService: UserService) {

    @GetMapping("/{id}")
    fun getUser(@PathVariable id: String): ResponseEntity<*> {
        val user = userService.getUser(id)
        return if (user != null) {
            ResponseEntity.ok(user)
        } else {
            ResponseEntity.notFound().build<Void>()
        }
    }

    @PostMapping
    fun createUser(@RequestBody user: User): ResponseEntity<String> {
        return if (userService.saveUser(user)) {
            ResponseEntity.created(URI.create("/api/users/${user.id}")).build()
        } else {
            ResponseEntity.badRequest().body("Failed to create user")
        }
    }

    // Kotlin特有的端点
    @GetMapping("/adults")
    fun getAdults(): ResponseEntity<List<User>> {
        if (userService is KotlinUserService) {
            return ResponseEntity.ok(userService.findAdults())
        }
        return ResponseEntity.ok(userService.getAllUsers().filter { it.isAdult() })
    }

    @GetMapping("/statistics")
    fun getStatistics(): ResponseEntity<Map<String, Int>> {
        if (userService is KotlinUserService) {
            return ResponseEntity.ok(userService.countByAgeGroup())
        }
        return ResponseEntity.ok(emptyMap())
    }
}
```

### 6.7.2 混合项目配置

```kotlin
// build.gradle.kts - 混合项目配置
plugins {
    id("org.springframework.boot") version "3.0.0"
    id("io.spring.dependency-management") version "1.1.0"
    kotlin("jvm") version "1.8.0"
    kotlin("plugin.spring") version "1.8.0"
}

dependencies {
    // Java依赖
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("org.springframework.boot:spring-boot-starter-data-jpa")

    // Kotlin支持
    implementation("org.jetbrains.kotlin:kotlin-reflect")
    implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin")

    // 协程支持
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-reactor")

    // 测试依赖
    testImplementation("org.springframework.boot:spring-boot-starter-test")
    testImplementation("org.jetbrains.kotlin:kotlin-test")
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit")
}

// 配置Kotlin编译选项
tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    kotlinOptions {
        freeCompilerArgs = listOf("-Xjsr305=strict")
        jvmTarget = "17"
    }
}

// Java配置
tasks.withType<JavaCompile> {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

// 配置Kotlin与Java的源码位置
sourceSets {
    main {
        java {
            srcDir("src/main/java")
        }
        kotlin {
            srcDir("src/main/kotlin")
        }
    }
    test {
        java {
            srcDir("src/test/java")
        }
        kotlin {
            srcDir("src/test/kotlin")
        }
    }
}
```

### 6.7.3 迁移实战案例

```kotlin
// 迁移实战：支付系统升级

// 1. 原Java支付接口
interface PaymentProcessor {
    @Throws(PaymentException::class)
    fun processPayment(payment: PaymentRequest): PaymentResult
}

data class PaymentRequest(
    val amount: Double,
    val currency: String,
    val cardNumber: String,
    val expiryDate: String
)

data class PaymentResult(
    val success: Boolean,
    val transactionId: String?,
    val errorMessage: String?
)

class PaymentException(message: String) : Exception(message)

// 2. Kotlin扩展实现
class KotlinPaymentProcessor : PaymentProcessor {

    // 实现Java接口
    override fun processPayment(payment: PaymentRequest): PaymentResult {
        return try {
            validatePayment(payment)
            val transactionId = executePayment(payment)
            PaymentResult(true, transactionId, null)
        } catch (e: PaymentException) {
            PaymentResult(false, null, e.message)
        }
    }

    // Kotlin特有的扩展方法
    private fun validatePayment(payment: PaymentRequest) {
        require(payment.amount > 0) { "Amount must be positive" }
        require(payment.cardNumber.matches(Regex("\\d{16}"))) { "Invalid card number" }
        require(payment.expiryDate.matches(Regex("\\d{2}/\\d{2}"))) { "Invalid expiry date" }
    }

    private fun executePayment(payment: PaymentRequest): String {
        // 模拟支付处理
        return "TXN_${System.currentTimeMillis()}"
    }

    // Kotlin特有的便利方法
    fun processPaymentAsync(payment: PaymentRequest): Deferred<PaymentResult> = CoroutineScope(Dispatchers.IO).async {
        processPayment(payment)
    }

    fun processBatchPayments(payments: List<PaymentRequest>): List<PaymentResult> {
        return runBlocking {
            payments.map { payment ->
                async { processPayment(payment) }
            }.awaitAll()
        }
    }
}

// 3. 混合使用的服务类
class OrderService {
    private val paymentProcessor: PaymentProcessor = KotlinPaymentProcessor()

    // Java风格的同步方法
    fun processOrderPayment(order: Order): PaymentResult {
        val payment = PaymentRequest(
            amount = order.total,
            currency = order.currency,
            cardNumber = order.cardNumber,
            expiryDate = order.expiryDate
        )
        return paymentProcessor.processPayment(payment)
    }

    // Kotlin风格的异步方法
    suspend fun processOrderPaymentAsync(order: Order): PaymentResult {
        val payment = PaymentRequest(
            amount = order.total,
            currency = order.currency,
            cardNumber = order.cardNumber,
            expiryDate = order.expiryDate
        )

        return if (paymentProcessor is KotlinPaymentProcessor) {
            paymentProcessor.processPaymentAsync(payment).await()
        } else {
            paymentProcessor.processPayment(payment)
        }
    }

    // 批量处理
    fun processBatchOrders(orders: List<Order>): List<PaymentResult> {
        val payments = orders.map { order ->
            PaymentRequest(
                amount = order.total,
                currency = order.currency,
                cardNumber = order.cardNumber,
                expiryDate = order.expiryDate
            )
        }

        return if (paymentProcessor is KotlinPaymentProcessor) {
            paymentProcessor.processBatchPayments(payments)
        } else {
            payments.map { paymentProcessor.processPayment(it) }
        }
    }
}

data class Order(
    val id: String,
    val total: Double,
    val currency: String,
    val cardNumber: String,
    val expiryDate: String
)

// 4. 测试类展示互操作性
class PaymentProcessorTest {
    private val paymentProcessor = KotlinPaymentProcessor()

    @Test
    fun testJavaStylePayment() {
        val payment = PaymentRequest(
            amount = 100.0,
            currency = "USD",
            cardNumber = "4111111111111111",
            expiryDate = "12/25"
        )

        val result = paymentProcessor.processPayment(payment)
        assertTrue(result.success)
        assertNotNull(result.transactionId)
    }

    @Test
    fun testKotlinStyleAsyncPayment() = runBlocking {
        val payment = PaymentRequest(
            amount = 200.0,
            currency = "EUR",
            cardNumber = "5555555555554444",
            expiryDate = "06/26"
        )

        val processor = paymentProcessor as KotlinPaymentProcessor
        val result = processor.processPaymentAsync(payment).await()
        assertTrue(result.success)
    }
}

// 使用示例
fun migrationDemo() {
    val orderService = OrderService()

    // Java风格调用
    val order1 = Order("1", 150.0, "USD", "4111111111111111", "12/25")
    val result1 = orderService.processOrderPayment(order1)
    println("Java风格结果: $result1")

    // Kotlin风格调用
    runBlocking {
        val order2 = Order("2", 200.0, "EUR", "5555555555554444", "06/26")
        val result2 = orderService.processOrderPaymentAsync(order2)
        println("Kotlin风格结果: $result2")
    }

    // 批量处理
    val orders = listOf(
        Order("3", 100.0, "USD", "4111111111111111", "12/25"),
        Order("4", 75.0, "EUR", "5555555555554444", "06/26"),
        Order("5", 125.0, "GBP", "378282246310005", "09/27")
    )
    val batchResults = orderService.processBatchOrders(orders)
    println("批量处理结果: $batchResults")
}
```

---

## 6.8 最佳实践和注意事项

### 6.8.1 性能优化建议

```kotlin
// 性能优化建议
fun performanceOptimizationTips() {
    // 1. 避免不必要的集合转换
    val javaList = java.util.ArrayList<Int>()
    repeat(1000) { javaList.add(it) }

    // 避免：每次都转换
    val sum = javaList.toList().filter { it % 2 == 0 }.sum()
    println("避免的方式: $sum")

    // 推荐：直接在Java集合上操作
    val efficientSum = javaList.asSequence()
        .filter { it % 2 == 0 }
        .sum()
    println("推荐的方式: $efficientSum")

    // 2. 合理使用平台类型
    fun processJavaString(javaString: String!): Int {
        // 使用安全调用处理可能的null
        return javaString?.length ?: 0
    }

    // 3. 内联函数优化
    inline fun <T> Collection<T>.measureTime(operation: (T) -> Unit): Long {
        val startTime = System.nanoTime()
        this.forEach(operation)
        return System.nanoTime() - startTime
    }

    val time = javaList.measureTime { /* 处理逻辑 */ }
    println("处理时间: ${time / 1_000_000}ms")

    // 4. 避免反射调用
    // 使用@JvmStatic或@JvmField来避免反射
    object Optimized {
        @JvmStatic
        fun fastMethod(): String = "Fast access"

        @JvmField
        val constant = 42
    }

    println("快速访问: ${Optimized.fastMethod()}")
    println("常量访问: ${Optimized.constant}")
}
```

### 6.8.2 代码风格统一

```kotlin
// 代码风格统一指南
object CodeStyleGuidelines {

    // 1. 命名约定
    const val CONSTANT_NAME = "UPPER_SNAKE_CASE"

    private var mutableProperty = 0

    fun functionName(): String {
        return "camelCase"
    }

    // 2. 注解使用
    data class User(
        @SerializedName("user_id") val id: String,
        @SerializedName("user_name") val name: String
    )

    // 3. 可见性控制
    private fun privateFunction() {}
    internal fun internalFunction() {}
    public fun publicFunction() {}

    // 4. 异常处理
    fun safeOperation(): Result<String> {
        return try {
            Result.success("Success")
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // 5. 文档注释
    /**
     * 计算两个数的和
     * @param a 第一个数
     * @param b 第二个数
     * @return 两数之和
     * @throws IllegalArgumentException 如果参数为null
     */
    @Throws(IllegalArgumentException::class)
    fun add(a: Int?, b: Int?): Int {
        requireNotNull(a) { "参数a不能为null" }
        requireNotNull(b) { "参数b不能为null" }
        return a + b
    }
}
```

### 6.8.3 迁移检查清单

```kotlin
// 迁移检查清单
object MigrationChecklist {

    fun runMigrationChecks(): List<String> {
        val issues = mutableListOf<String>()

        // 1. 检查空安全
        issues.addAll(checkNullSafety())

        // 2. 检查异常处理
        issues.addAll(checkExceptionHandling())

        // 3. 检查集合使用
        issues.addAll(checkCollectionUsage())

        // 4. 检查性能问题
        issues.addAll(checkPerformanceIssues())

        return issues
    }

    private fun checkNullSafety(): List<String> {
        val issues = mutableListOf<String>()

        // 模拟检查代码中的空安全问题
        // 实际实现中会使用静态分析工具

        return issues
    }

    private fun checkExceptionHandling(): List<String> {
        val issues = mutableListOf<String>()

        // 检查是否有未处理的检查异常

        return issues
    }

    private fun checkCollectionUsage(): List<String> {
        val issues = mutableListOf<String>()

        // 检查是否有不必要的集合转换

        return issues
    }

    private fun checkPerformanceIssues(): List<String> {
        val issues = mutableListOf<String>()

        // 检查是否有反射调用等性能问题

        return issues
    }
}

// 迁移工具类
class MigrationHelper {

    companion object {
        /**
         * 将Java POJO转换为Kotlin data class
         */
        fun generateDataClass(javaClass: Class<*>): String {
            val fields = javaClass.declaredFields
                .filter { !Modifier.isStatic(it.modifiers) }
                .joinToString(",\n        ") { field ->
            """${field.name}: ${field.type.simpleName}"""
        }

            return """
@JvmName("create${javaClass.simpleName}")
data class ${javaClass.simpleName}(
        $fields
            )""".trimIndent()
        }

        /**
         * 生成Kotlin扩展函数
         */
        fun generateExtensionFunctions(javaClass: Class<*>): List<String> {
            val functions = mutableListOf<String>()

            // 为所有getter生成扩展函数
            javaClass.methods
                .filter { it.name.startsWith("get") && it.parameterCount == 0 }
                .forEach { method ->
                    val propertyName = method.name.substring(3).replaceFirstChar { it.lowercase() }
                    val returnType = method.returnType.simpleName
                    functions.add(
                        "fun ${javaClass.simpleName}.$propertyName(): $returnType = this.${method.name}()"
                    )
                }

            return functions
        }

        /**
         * 检查Java类的Kotlin兼容性
         */
        fun checkKotlinCompatibility(javaClass: Class<*>): CompatibilityReport {
            val issues = mutableListOf<String>()

            // 检查字段名冲突
            javaClass.declaredFields.forEach { field ->
                if (field.name in listOf("getClass", "hashCode", "toString")) {
                    issues.add("字段名冲突: ${field.name}")
                }
            }

            // 检查方法重载冲突
            val methodSignatures = javaClass.methods
                .groupBy { "${it.name}-${it.parameterCount}" }
                .filter { it.value.size > 1 }
                .keys

            if (methodSignatures.isNotEmpty()) {
                issues.add("方法重载可能的问题: $methodSignatures")
            }

            return CompatibilityReport(
                className = javaClass.simpleName,
                issues = issues,
                isCompatible = issues.isEmpty()
            )
        }
    }

    data class CompatibilityReport(
        val className: String,
        val issues: List<String>,
        val isCompatible: Boolean
    )
}
```

---

## 6.9 本章小结

### ✅ 核心概念掌握

通过本章学习，您已经掌握了Kotlin与Java互操作的完整体系：

1. **Kotlin调用Java代码**
   - 基础互操作语法和规则
   - Java可空性的处理策略
   - Java集合与Kotlin集合的转换
   - SAM转换的简化语法

2. **Java调用Kotlin代码**
   - Kotlin代码的Java可见性规则
   - @JvmField、@JvmStatic、@JvmOverloads的使用
   - 顶层函数和object的访问方式
   - data class的Java兼容性

3. **注解处理兼容性**
   - KAPT vs KSP的对比和选择
   - Lombok集成的最佳实践
   - 自定义注解的使用
   - 元数据处理的注意事项

4. **函数式接口的互操作**
   - Java函数式接口的Kotlin实现
   - SAM转换的应用场景
   - 自定义函数式接口的设计
   - Lambda表达式的桥接

5. **实战迁移策略**
   - 渐进式迁移的实施步骤
   - 混合项目的配置管理
   - 性能优化建议
   - 代码风格的统一

### ✅ 互操作的优势

| 特性 | 纯Java项目 | 纯Kotlin项目 | 混合项目 | 优势说明 |
|------|-----------|-------------|----------|----------|
| 迁移成本 | 无 | 完全重写 | 渐进式 | ⭐⭐⭐⭐⭐ |
| 团队学习成本 | 低 | 高 | 中等 | ⭐⭐⭐⭐ |
| 性能优化 | 有限 | 优秀 | 逐步提升 | ⭐⭐⭐⭐ |
| 代码复用 | 一般 | 良好 | 最佳 | ⭐⭐⭐⭐⭐ |
| 风险控制 | 低 | 中 | 低 | ⭐⭐⭐⭐ |

### ✅ 实战要点

1. **迁移策略**
   - 从工具类开始，逐步迁移核心业务逻辑
   - 保持API接口的向后兼容性
   - 充分利用Kotlin的协程和函数式特性
   - 建立代码审查和质量检查机制

2. **性能优化**
   - 避免不必要的集合转换
   - 合理使用平台类型
   - 优先使用内联函数
   - 减少反射调用

3. **团队协作**
   - 建立统一的编码规范
   - 提供Kotlin培训和支持
   - 使用自动化工具检查互操作问题
   - 鼓励分享最佳实践

### 📚 下一步学习

下一章我们将探索**Kotlin在Android开发中的实践**，包括：
- Android项目的Kotlin配置
- ViewBinding与属性委托
- ViewModel与LiveData的Kotlin优化
- 协程在Android中的应用
- Jetpack Compose基础
- 架构模式与依赖注入

这将帮助您在Android开发中充分发挥Kotlin的优势！

---

## 📝 章节练习

### 基础练习
1. 将以下Java POJO转换为Kotlin data class：
```java
public class Person {
    private String name;
    private int age;
    private String email;

    public Person(String name, int age, String email) {
        this.name = name;
        this.age = age;
        this.email = email;
    }

    // getters and setters...
    // equals, hashCode, toString...
}
```

2. 创建一个Kotlin工具类，提供常用的字符串处理方法，并确保能在Java中方便调用。

### 进阶练习
1. 实现一个混合项目配置：
   - 支持Java和Kotlin源码共存
   - 配置正确的编译依赖
   - 处理注解处理器的兼容性
   - 编写自动化迁移脚本

2. 创建一个兼容层，让Java代码能够使用Kotlin的协程和Flow功能。

### 挑战练习
1. 构建一个完整的迁移工具：
   - 自动检测Java代码中的可改进点
   - 生成对应的Kotlin代码
   - 提供迁移报告和建议
   - 支持批量转换和验证

2. 设计一个互操作测试框架：
   - 自动生成Java-Kotlin互操作测试用例
   - 验证类型转换的正确性
   - 性能基准测试
   - 异常处理验证

---

**恭喜完成Kotlin与Java互操作的学习！您现在已经掌握了在现有Java项目中成功引入和集成Kotlin的所有技能，能够平滑地进行项目升级和技术栈转型！**