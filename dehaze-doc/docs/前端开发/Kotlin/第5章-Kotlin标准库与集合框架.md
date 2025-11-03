# 第5章：Kotlin标准库与集合框架

## 📖 章节概述

Kotlin标准库提供了丰富的工具函数和集合框架，相比Java有了显著的改进和增强。本章将深入探讨Kotlin标准库的核心功能，包括集合类型、范围操作、扩展函数、时间处理等。作为Java开发者，您将学习到如何利用这些强大的工具来编写更简洁、更高效的代码。

**学习时长**: 约3-4天
**核心目标**: 熟练掌握Kotlin标准库和集合框架，能够高效处理各种数据操作

---

## 5.1 Kotlin标准库概览

### 5.1.1 标准库结构组成

```kotlin
// Kotlin标准库的主要组成部分
/*
kotlin-stdlib/
├── kotlin/
│   ├── collections/     // 集合框架
│   ├── ranges/          // 范围操作
│   ├── sequences/       // 序列操作
│   ├── text/            // 字符串处理
│   ├── time/            // 时间处理
│   ├── reflect/         // 反射工具
│   └── ...              // 其他工具类
└── kotlinx/
    ├── coroutines/      // 协程库
    ├── serialization/   // 序列化库
    └── ...              // 其他扩展库
*/

// 标准库核心包结构
import kotlin.collections.*  // 集合操作
import kotlin.ranges.*        // 范围相关
import kotlin.text.*          // 字符串操作
import kotlin.math.*          // 数学运算
import kotlin.random.*        // 随机数生成
```

### 5.1.2 常用工具类和函数

```kotlin
// 数学运算工具
fun mathOperationsDemo() {
    println("=== 数学运算工具 ===")

    // 基础数学函数
    println("绝对值: ${abs(-5.3)}")                    // 5.3
    println("最大值: ${maxOf(3, 7, 2, 9)}")            // 9
    println("最小值: ${minOf(3, 7, 2, 9)}")            // 2
    println("平方根: ${sqrt(16.0)}")                   // 4.0
    println("幂运算: ${pow(2.0, 3.0)}")                // 8.0
    println("向上取整: ${ceil(3.2)}")                  // 4.0
    println("向下取整: ${floor(3.8)}")                 // 3.0
    println("四舍五入: ${round(3.6)}")                 // 4

    // 三角函数
    println("正弦值: ${sin(Math.PI / 2)}")             // 1.0
    println("余弦值: ${cos(0.0)}")                    // 1.0

    // 随机数生成
    val random = Random(42) // 固定种子的随机数生成器
    println("随机整数: ${random.nextInt(10)}")         // 0-9的随机数
    println("随机双精度: ${random.nextDouble()}")      // 0.0-1.0的随机数
    println("随机布尔: ${random.nextBoolean()}")       // 随机布尔值

    // 随机数工具函数
    val randomValues = (1..100).random()             // 1-100的随机数
    val randomChoice = listOf("A", "B", "C").random() // 随机选择元素
    println("随机值: $randomValues, 随机选择: $randomChoice")
}

// 类型转换工具
fun typeConversionDemo() {
    println("\n=== 类型转换工具 ===")

    // 安全的类型转换
    val strNumber = "123"
    val safeInt = strNumber.toIntOrNull()
    println("安全整数转换: $safeInt")                // 123

    val invalidNumber = "abc"
    val nullInt = invalidNumber.toIntOrNull()
    println("无效数字转换: $nullInt")                // null

    // 字符串到其他类型
    val strDouble = "3.14"
    val doubleValue = strDouble.toDoubleOrNull()
    println("字符串转双精度: $doubleValue")          // 3.14

    // 数值到字符串
    val number = 42
    val strNumber2 = number.toString()
    println("数字转字符串: $strNumber2")              // "42"

    // 进制转换
    val binary = "1010"
    val decimal = binary.toInt(2)
    println("二进制转十进制: $decimal")              // 10

    val hex = "FF"
    val decimal2 = hex.toInt(16)
    println("十六进制转十进制: $decimal2")            // 255
}

// 条件和比较工具
fun conditionalToolsDemo() {
    println("\n=== 条件和比较工具 ===")

    // 三元操作符的替代
    val a = 5, b = 3
    val max = if (a > b) a else b
    println("最大值: $max")                          // 5

    // 值的范围检查
    val score = 85
    val grade = when {
        score >= 90 -> "A"
        score >= 80 -> "B"
        score >= 70 -> "C"
        score >= 60 -> "D"
        else -> "F"
    }
    println("成绩等级: $grade")                      // B

    // 使用in操作符的范围检查
    val age = 25
    val isAdult = age in 18..65
    println("是否成年: $isAdult")                    // true

    // 集合包含检查
    val fruits = listOf("apple", "banana", "orange")
    val hasApple = "apple" in fruits
    println("是否有苹果: $hasApple")                 // true
}
```

---

## 5.2 集合类型详解

### 5.2.1 List集合详解

```kotlin
// List的创建和基本操作
fun listBasicsDemo() {
    println("=== List基础操作 ===")

    // 创建List
    val immutableList = listOf("Apple", "Banana", "Orange")
    val mutableList = mutableListOf("Red", "Green", "Blue")

    // 访问元素
    println("第一个元素: ${immutableList.first()}")
    println("最后一个元素: ${immutableList.last()}")
    println("索引1的元素: ${immutableList[1]}")

    // 安全访问
    println("索引5的元素(安全): ${immutableList.getOrNull(5)}")
    println("索引5的元素(默认): ${immutableList.getOrElse(5) { "默认值" }}")

    // List的修改操作（仅限MutableList）
    mutableList.add("Yellow")
    mutableList.add(1, "Purple")  // 在指定位置插入
    mutableList[0] = "Crimson"    // 修改指定位置元素
    mutableList.removeAt(2)       // 移除指定位置元素

    println("修改后的List: $mutableList")

    // List的查找操作
    val numbers = listOf(1, 2, 3, 2, 4, 5, 2)
    println("第一个2的索引: ${numbers.indexOf(2)}")          // 1
    println("最后一个2的索引: ${numbers.lastIndexOf(2)}")     // 6
    println("是否包含3: ${3 in numbers}")                    // true
    println("子列表位置: ${numbers.indexOfSubList(listOf(2, 3))}") // 1

    // List的切片操作
    val subList = numbers.subList(1, 5) // 索引1到4（不包含5）
    println("子列表: $subList")                              // [2, 3, 2, 4]

    val slice = numbers.slice(2..4)                          // 索引2到4
    println("切片: $slice")                                  // [3, 2, 4]
}

// List的高级操作
fun listAdvancedDemo() {
    println("\n=== List高级操作 ===")

    val products = listOf(
        Product("Laptop", 1200.0, "Electronics"),
        Product("Book", 25.0, "Education"),
        Product("Phone", 800.0, "Electronics"),
        Product("Pen", 2.0, "Office"),
        Product("Tablet", 600.0, "Electronics")
    )

    // 排序操作
    val sortedByName = products.sortedBy { it.name }
    println("按名称排序: ${sortedByName.map { it.name }}")

    val sortedByPriceDesc = products.sortedByDescending { it.price }
    println("按价格降序: ${sortedByPriceDesc.map { "${it.name}:${it.price}" }}")

    val customSorted = products.sortedWith(compareBy<Product> { it.category }
        .thenBy { it.price })
    println("自定义排序: ${customSorted.map { "${it.category}-${it.name}" }}")

    // 分组操作
    val groupedByCategory = products.groupBy { it.category }
    println("按分类分组:")
    groupedByCategory.forEach { (category, items) ->
        println("  $category: ${items.map { it.name }}")
    }

    // 分区操作
    val (expensive, cheap) = products.partition { it.price > 500.0 }
    println("昂贵商品: ${expensive.map { it.name }}")
    println("便宜商品: ${cheap.map { it.name }}")

    // 统计操作
    val totalPrice = products.sumOf { it.price }
    val averagePrice = products.averageOf { it.price }
    val maxPriceProduct = products.maxByOrNull { it.price }
    val expensiveCount = products.count { it.price > 100.0 }

    println("总价格: $totalPrice")
    println("平均价格: $averagePrice")
    println("最贵商品: ${maxPriceProduct?.name}")
    println("昂贵商品数量: $expensiveCount")

    // 去重操作
    val numbers = listOf(1, 2, 3, 2, 4, 1, 5)
    val distinctNumbers = numbers.distinct()
    println("去重后: $distinctNumbers")

    val distinctByLength = listOf("hello", "world", "kotlin", "java")
        .distinctBy { it.length }
    println("按长度去重: $distinctByLength")
}

data class Product(val name: String, val price: Double, val category: String)
```

### 5.2.2 Set集合详解

```kotlin
// Set的特性和操作
fun setBasicsDemo() {
    println("=== Set基础操作 ===")

    // 创建Set
    val immutableSet = setOf("Apple", "Banana", "Orange")
    val mutableSet = mutableSetOf(1, 2, 3, 4, 5)

    // Set的特性：自动去重
    val duplicateSet = setOf(1, 2, 2, 3, 3, 3, 4)
    println("自动去重: $duplicateSet")                 // [1, 2, 3, 4]

    // 基本操作
    println("Set大小: ${immutableSet.size}")
    println("是否为空: ${immutableSet.isEmpty()}")
    println("是否包含Apple: ${"Apple" in immutableSet}")
    println("是否包含Grape: ${"Grape" in immutableSet}")

    // MutableSet的修改操作
    mutableSet.add(6)
    mutableSet.addAll(setOf(7, 8, 9))
    mutableSet.remove(3)
    mutableSet.removeAll(setOf(1, 2))
    mutableSet.retainAll(setOf(4, 5, 6, 7, 8, 9))

    println("修改后的MutableSet: $mutableSet")

    // 集合运算
    val setA = setOf(1, 2, 3, 4, 5)
    val setB = setOf(4, 5, 6, 7, 8)

    println("SetA: $setA")
    println("SetB: $setB")

    // 交集
    val intersection = setA intersect setB
    println("交集: $intersection")                      // [4, 5]

    // 并集
    val union = setA union setB
    println("并集: $union")                             // [1, 2, 3, 4, 5, 6, 7, 8]

    // 差集
    val difference = setA subtract setB
    println("差集 (A-B): $difference")                  // [1, 2, 3]

    val difference2 = setB subtract setA
    println("差集 (B-A): $difference2")                 // [6, 7, 8]

    // 对称差集
    val symmetricDifference = (setA subtract setB) union (setB subtract setA)
    println("对称差集: $symmetricDifference")          // [1, 2, 3, 6, 7, 8]
}

// SortedSet和NavigableSet的使用
fun sortedSetDemo() {
    println("\n=== SortedSet操作 ===")

    // SortedSet自动排序
    val sortedSet = sortedSetOf(5, 2, 8, 1, 9, 3)
    println("自动排序: $sortedSet")                     // [1, 2, 3, 5, 8, 9]

    // 自定义排序
    val customSortedSet = sortedSetOf(compareByDescending { it }, 10, 5, 15, 8)
    println("自定义排序: $customSortedSet")             // [15, 10, 8, 5]

    // TreeSet操作
    val treeSet = TreeSet<String>()
    treeSet.addAll(listOf("Zebra", "Apple", "Banana", "Cherry"))
    println("TreeSet: $treeSet")                        // [Apple, Banana, Cherry, Zebra]

    // TreeSet特有的操作
    println("第一个元素: ${treeSet.first()}")             // Apple
    println("最后一个元素: ${treeSet.last()}")            // Zebra

    // 子集操作
    val headSet = treeSet.headSet("Cherry")             // 不包含Cherry
    println("头子集: $headSet")                         // [Apple, Banana]

    val tailSet = treeSet.tailSet("Banana")              // 包含Banana
    println("尾子集: $tailSet")                         // [Banana, Cherry, Zebra]

    val subSet = treeSet.subSet("Banana", "Cherry")     // 包含Banana，不包含Cherry
    println("子集: $subSet")                            // [Banana]
}

// Set的实际应用场景
fun setPracticalDemo() {
    println("\n=== Set实际应用 ===")

    // 场景1：权限管理
    val allPermissions = setOf("READ", "WRITE", "DELETE", "ADMIN")
    val userPermissions = mutableSetOf("READ", "WRITE")

    println("用户权限: $userPermissions")
    println("是否有删除权限: ${"DELETE" in userPermissions}")

    // 授予权限
    userPermissions.add("DELETE")
    println("授予删除权限后: $userPermissions")

    // 场景2：标签系统
    val articleTags = mutableSetOf("kotlin", "android", "programming")
    val requiredTags = setOf("kotlin", "java")

    println("文章标签: $articleTags")
    println("是否包含所需标签: ${requiredTags.all { it in articleTags }}")

    // 场景3：去重处理
    val rawInput = listOf("user1", "user2", "user1", "user3", "user2", "user4")
    val uniqueUsers = rawInput.toSet().toList()
    println("原始输入: $rawInput")
    println("去重后: $uniqueUsers")

    // 场景4：查找重复项
    val findDuplicates = { list: List<String> ->
        val seen = mutableSetOf<String>()
        val duplicates = mutableSetOf<String>()

        list.forEach { item ->
            if (!seen.add(item)) {
                duplicates.add(item)
            }
        }
        duplicates.toList()
    }

    val duplicateUsers = findDuplicates(rawInput)
    println("重复用户: $duplicateUsers")
}
```

### 5.2.3 Map集合详解

```kotlin
// Map的创建和基本操作
fun mapBasicsDemo() {
    println("=== Map基础操作 ===")

    // 创建Map
    val immutableMap = mapOf(
        "name" to "张三",
        "age" to 25,
        "city" to "北京"
    )

    val mutableMap = mutableMapOf(
        "product1" to 100.0,
        "product2" to 200.0,
        "product3" to 150.0
    )

    // 访问元素
    println("姓名: ${immutableMap["name"]}")              // 张三
    println("年龄: ${immutableMap.get("age")}")          // 25

    // 安全访问
    println("不存在键的值: ${immutableMap.getOrDefault("country", "中国")}")
    println("不存在的键值: ${immutableMap.getOrElse("country") { "未知" }}")

    // 检查键值存在性
    println("是否包含name键: ${"name" in immutableMap}")
    println("是否包含张三: ${"张三" in immutableMap.values}")

    // MutableMap的修改操作
    mutableMap["product4"] = 300.0                        // 添加或更新
    mutableMap.put("product5", 250.0)                     // 添加
    mutableMap.remove("product2")                         // 删除
    mutableMap.replace("product3", 180.0)                 // 替换

    println("修改后的Map: $mutableMap")

    // 批量操作
    mutableMap.putAll(mapOf(
        "product6" to 400.0,
        "product7" to 350.0
    ))

    mutableMap.keys.removeAll { key ->
        key.endsWith("1") || key.endsWith("2")
    }

    println("批量操作后: $mutableMap")
}

// Map的高级操作
fun mapAdvancedDemo() {
    println("\n=== Map高级操作 ===")

    val students = mapOf(
        "001" to Student("张三", 85, "数学"),
        "002" to Student("李四", 92, "英语"),
        "003" to Student("王五", 78, "数学"),
        "004" to Student("赵六", 88, "物理"),
        "005" to Student("钱七", 95, "英语")
    )

    // 过滤操作
    val excellentStudents = students.filter { (_, student) ->
        student.score >= 90
    }
    println("优秀学生: ${excellentStudents.map { "${it.value.name}:${it.value.score}" }}")

    val mathStudents = students.filterValues { student ->
        student.subject == "数学"
    }
    println("数学学生: ${mathStudents.map { "${it.value.name}" }}")

    // 映射操作
    val studentNames = students.mapValues { student ->
        "姓名: ${student.value.name}, 分数: ${student.value.score}"
    }
    println("学生信息: $studentNames")

    val nameToScore = students.mapKeys { (id, _) ->
        "学号$id"
    }.mapValues { it.value.score }
    println("姓名到分数: $nameToScore")

    // 分组操作
    val studentsBySubject = students.entries
        .groupBy { entry ->
            entry.value.subject
        }
        .mapValues { (_, entries) ->
            entries.map { it.value.name }
        }
    println("按科目分组: $studentsBySubject")

    // 统计操作
    val averageScore = students.values.averageOf { it.score }
    val maxScore = students.values.maxOfOrNull { it.score }
    val subjectCounts = students.values.groupBy { it.subject }
        .mapValues { (_, students) -> students.size }

    println("平均分: $averageScore")
    println("最高分: $maxScore")
    println("各科目人数: $subjectCounts")

    // 查找操作
    val firstMathStudent = students.values.find { it.subject == "数学" }
    println("第一个数学学生: $firstMathStudent")

    val allPassStudents = students.filter { (_, student) ->
        student.score >= 60
    }
    println("及格学生数: ${allPassStudents.size}")
}

data class Student(val name: String, val score: Int, val subject: String)

// Map的实际应用场景
fun mapPracticalDemo() {
    println("\n=== Map实际应用 ===")

    // 场景1：配置管理
    val appConfig = mutableMapOf(
        "server.url" to "https://api.example.com",
        "server.timeout" to "5000",
        "cache.enabled" to "true",
        "cache.maxSize" to "100"
    )

    fun getConfig(key: String): String? = appConfig[key]
    fun setConfig(key: String, value: String) {
        appConfig[key] = value
    }

    println("服务器URL: ${getConfig("server.url")}")
    setConfig("server.retries", "3")
    println("重试次数: ${getConfig("server.retries")}")

    // 场景2：数据聚合
    val salesData = listOf(
        Sale("产品A", "北京", 1000.0),
        Sale("产品B", "上海", 1500.0),
        Sale("产品A", "广州", 800.0),
        Sale("产品C", "北京", 1200.0),
        Sale("产品B", "深圳", 1800.0)
    )

    val salesByProduct = salesData
        .groupBy { it.product }
        .mapValues { (_, sales) ->
            sales.sumOf { it.amount }
        }

    val salesByRegion = salesData
        .groupBy { it.region }
        .mapValues { (_, sales) ->
            Triple(
                sales.size,
                sales.sumOf { it.amount },
                sales.averageOf { it.amount }
            )
        }

    println("产品销售额:")
    salesByProduct.forEach { (product, total) ->
        println("  $product: ¥$total")
    }

    println("地区销售统计:")
    salesByRegion.forEach { (region, stats) ->
        val (count, total, average) = stats
        println("  $region: $count笔, 总计¥$total, 平均¥$average")
    }

    // 场景3：计数器
    val wordCounter = mutableMapOf<String, Int>()

    val text = "kotlin is great kotlin is powerful kotlin is concise"
    text.split(" ").forEach { word ->
        wordCounter[word] = wordCounter.getOrDefault(word, 0) + 1
    }

    println("词频统计:")
    wordCounter.entries
        .sortedByDescending { it.value }
        .forEach { (word, count) ->
            println("  $word: $count次")
        }

    // 场景4：缓存实现
    class SimpleCache<K, V> {
        private val cache = mutableMapOf<K, V>()
        private val maxSize = 100

        fun get(key: K): V? = cache[key]

        fun put(key: K, value: V) {
            if (cache.size >= maxSize) {
                // 简单的LRU：移除第一个元素
                cache.entries.firstOrNull()?.let {
                    cache.remove(it.key)
                }
            }
            cache[key] = value
        }

        fun remove(key: K): V? = cache.remove(key)
        fun clear() = cache.clear()
        fun size() = cache.size
        fun keys() = cache.keys.toList()
    }

    val cache = SimpleCache<String, String>()
    cache.put("user:1", "张三")
    cache.put("user:2", "李四")
    cache.put("user:3", "王五")

    println("缓存大小: ${cache.size()}")
    println("用户1: ${cache.get("user:1")}")
    println("缓存键: ${cache.keys()}")
}

data class Sale(val product: String, val region: String, val amount: Double)
```

---

## 5.3 范围与序列

### 5.3.1 Range（范围）详解

```kotlin
// 基础范围操作
fun rangeBasicsDemo() {
    println("=== 基础范围操作 ===")

    // 整数范围
    val intRange = 1..10
    println("整数范围: $intRange")                     // 1..10

    val intRange2 = IntRange(1, 10)
    println("IntRange: $intRange2")                     // 1..10

    // 检查包含关系
    println("5在范围内: ${5 in intRange}")              // true
    println("10在范围内: ${10 in intRange}")            // true
    println("11在范围内: ${11 in intRange}")            // false
    println("0在范围内: ${0 in intRange}")              // false

    // 反向范围
    val reverseRange = 10 downTo 1
    println("反向范围: $reverseRange")                  // 10 downTo 1

    // 步长范围
    val stepRange = 1..10 step 2
    println("步长范围: $stepRange")                    // 1..10 step 2

    val evenNumbers = 2..20 step 2
    println("偶数范围: $evenNumbers")                   // 2..20 step 2

    // 开放范围（不包含结束值）
    val openRange = 1 until 10
    println("开放范围: $openRange")                     // 1 until 9

    // 字符范围
    val charRange = 'a'..'z'
    println("字符范围: $charRange")                     // a..z
    println("'m'在范围内: ${'m' in charRange}")         // true
    println("'A'在范围内: ${'A' in charRange}")         // false

    val upperCaseRange = 'A'..'Z'
    val digitRange = '0'..'9'

    // 范围的遍历
    println("整数遍历:")
    for (i in 1..5) {
        print("$i ")                                   // 1 2 3 4 5
    }
    println()

    println("反向遍历:")
    for (i in 5 downTo 1) {
        print("$i ")                                   // 5 4 3 2 1
    }
    println()

    println("步长遍历:")
    for (i in 0..10 step 2) {
        print("$i ")                                   // 0 2 4 6 8 10
    }
    println()
}

// 范围的高级操作
fun rangeAdvancedDemo() {
    println("\n=== 范围高级操作 ===")

    // 范围的属性
    val range = 1..10
    println("范围起始: ${range.first}")                 // 1
    println("范围结束: ${range.last}")                  // 10
    println("范围步长: ${range.step}")                  // 1
    println("范围是否为空: ${range.isEmpty()}")         // false

    val emptyRange = 5..1
    println("空范围: $emptyRange")                      // 5..1
    println("空范围是否为空: ${emptyRange.isEmpty()}")  // true

    // 范围的包含操作
    val letterRange = 'a'..'z'
    val testChars = listOf('a', 'm', 'z', 'A', '0')
    testChars.forEach { char ->
        println("$char 在字母范围内: ${char in letterRange}")
    }

    // 范围的转换
    val numbers = (1..10).toList()
    println("范围转列表: $numbers")

    val rangeFromList = numbers.first()..numbers.last()
    println("列表转范围: $rangeFromList")

    // 范围的交集
    fun intersectRanges(r1: IntRange, r2: IntRange): IntRange? {
        val start = maxOf(r1.first, r2.first)
        val end = minOf(r1.last, r2.last)
        return if (start <= end) start..end else null
    }

    val range1 = 1..10
    val range2 = 5..15
    val intersection = intersectRanges(range1, range2)
    println("范围交集: $intersection")                  // 5..10

    // 范围的应用：分数评级
    fun getGrade(score: Int): String {
        return when (score) {
            in 90..100 -> "A"
            in 80..89 -> "B"
            in 70..79 -> "C"
            in 60..69 -> "D"
            in 0..59 -> "F"
            else -> "无效分数"
        }
    }

    val scores = listOf(95, 82, 75, 68, 55, 105, -5)
    scores.forEach { score ->
        println("分数 $score: 等级 ${getGrade(score)}")
    }

    // 范围的应用：日期验证
    fun isValidDate(day: Int, month: Int): Boolean {
        val dayRange = when (month) {
            2 -> 1..28 // 简化处理，不考虑闰年
            in listOf(4, 6, 9, 11) -> 1..30
            in 1..12 -> 1..31
            else -> return false
        }
        return day in dayRange
    }

    val dates = listOf(Pair(15, 8), Pair(31, 4), Pair(29, 2), Pair(32, 1))
    dates.forEach { (day, month) ->
        val isValid = isValidDate(day, month)
        println("$month月$day日: ${if (isValid) "有效" else "无效"}")
    }
}
```

### 5.2.2 Sequence（序列）详解

```kotlin
// Sequence的基本概念
fun sequenceBasicsDemo() {
    println("=== Sequence基础操作 ===")

    // 创建Sequence
    val sequence1 = sequenceOf(1, 2, 3, 4, 5)
    println("Sequence: ${sequence1.toList()}")

    // 从集合创建Sequence
    val list = listOf(1, 2, 3, 4, 5)
    val sequence2 = list.asSequence()
    println("从List创建Sequence: ${sequence2.toList()}")

    // 使用generateSequence
    val sequence3 = generateSequence(1) { it + 1 }
        .take(10)
    println("generateSequence: ${sequence3.toList()}")

    // 使用yield构建Sequence
    val fibonacciSequence = sequence {
        var a = 0
        var b = 1
        while (true) {
            yield(a)
            val temp = a + b
            a = b
            b = temp
        }
    }
        .take(10)
    println("斐波那契Sequence: ${fibonacciSequence.toList()}")

    // Sequence与List的性能对比
    val largeList = (1..100000).toList()

    // List操作 - 创建中间集合
    val listStartTime = System.currentTimeMillis()
    val listResult = largeList
        .filter { it % 2 == 0 }           // 创建新List
        .map { it * it }                  // 创建新List
        .filter { it > 1000 }             // 创建新List
        .take(10)                         // 创建新List
        .sum()
    val listTime = System.currentTimeMillis() - listStartTime

    // Sequence操作 - 惰性计算
    val sequenceStartTime = System.currentTimeMillis()
    val sequenceResult = largeList.asSequence()
        .filter { it % 2 == 0 }           // 不创建中间集合
        .map { it * it }                  // 不创建中间集合
        .filter { it > 1000 }             // 不创建中间集合
        .take(10)                         // 不创建中间集合
        .sum()
    val sequenceTime = System.currentTimeMillis() - sequenceStartTime

    println("List操作结果: $listResult, 耗时: ${listTime}ms")
    println("Sequence操作结果: $sequenceResult, 耗时: ${sequenceTime}ms")
    println("性能提升: ${((listTime - sequenceTime).toDouble() / listTime * 100).toInt()}%")
}

// Sequence的高级操作
fun sequenceAdvancedDemo() {
    println("\n=== Sequence高级操作 ===")

    // 中间操作和终端操作
    val numbers = (1..10).asSequence()

    // 中间操作（惰性）
    val processedSequence = numbers
        .filter { it % 2 == 0 }         // 中间操作
        .map { it * it }                // 中间操作
        .filter { it > 10 }             // 中间操作

    println("中间操作后还未计算")

    // 终端操作（触发计算）
    val result = processedSequence.toList()
    println("终端操作后: $result")

    // Sequence的短路操作
    val infiniteSequence = generateSequence(1) { it + 1 }

    val first10 = infiniteSequence
        .filter { it % 2 == 0 }
        .map { it * it }
        .take(5)
        .toList()
    println("无限Sequence取前5个: $first10")

    // Sequence的查找操作
    val findResult = (1..1000).asSequence()
        .first { it % 7 == 0 && it % 13 == 0 }
    println("第一个能被7和13整除的数: $findResult")

    val findAnyResult = (1..100).asSequence()
        .find { it > 50 && it isPrime() }
    println("第一个大于50的质数: $findAnyResult")

    // Sequence的聚合操作
    val sumOfSquares = (1..100).asSequence()
        .filter { it % 3 == 0 }
        .map { it * it }
        .sum()
    println("3的倍数的平方和: $sumOfSquares")

    val averageOfOdds = (1..100).asSequence()
        .filter { it % 2 == 1 }
        .average()
    println("奇数的平均值: $averageOfOdds")
}

// 判断质数的辅助函数
fun Int.isPrime(): Boolean {
    if (this <= 1) return false
    if (this <= 3) return true
    if (this % 2 == 0 || this % 3 == 0) return false

    var i = 5
    while (i * i <= this) {
        if (this % i == 0 || this % (i + 2) == 0) return false
        i += 6
    }
    return true
}

// Sequence的实际应用
fun sequencePracticalDemo() {
    println("\n=== Sequence实际应用 ===")

    // 场景1：处理大型数据集
    val largeNumbers = (1..1000000).asSequence()

    val statistics = largeNumbers
        .filter { it % 2 == 0 }                    // 筛选偶数
        .take(10000)                               // 取前10000个
        .map { it.toDouble() }
        .let { numbers ->
            Triple(
                numbers.count().toLong(),
                numbers.average(),
                numbers.sum()
            )
        }

    val (count, average, sum) = statistics
    println("前10000个偶数统计:")
    println("  数量: $count")
    println("  平均值: $average")
    println("  总和: $sum")

    // 场景2：文件处理（模拟）
    fun processLogFile(lines: Sequence<String>): Map<String, Int> {
        return lines
            .filter { it.contains("ERROR") }
            .map { line ->
                val message = line.substringAfter("ERROR: ")
                message.trim()
            }
            .groupBy { it }
            .mapValues { (_, messages) -> messages.size }
    }

    val logLines = sequenceOf(
        "INFO: Application started",
        "ERROR: Database connection failed",
        "INFO: User logged in",
        "ERROR: Invalid user credentials",
        "ERROR: Database connection failed",
        "INFO: User logged out",
        "ERROR: File not found"
    )

    val errorCounts = processLogFile(logLines)
    println("错误统计:")
    errorCounts.forEach { (error, count) ->
        println("  $error: $count次")
    }

    // 场景3：流式数据处理
    data class Transaction(
        val id: String,
        val amount: Double,
        val category: String,
        val timestamp: Long
    )

    fun generateTransactions(): Sequence<Transaction> = sequence {
        var id = 1
        while (true) {
            yield(Transaction(
                id = "TXN$id",
                amount = (10..1000).random().toDouble(),
                category = listOf("Food", "Transport", "Shopping", "Bills").random(),
                timestamp = System.currentTimeMillis()
            ))
            id++
        }
    }

    val transactions = generateTransactions()
        .take(1000)                                // 取前1000笔交易
        .filter { it.amount > 100 }                // 筛选大额交易
        .groupBy { it.category }                   // 按分类分组
        .mapValues { (_, txns) ->
            Triple(
                txns.size,
                txns.sumOf { it.amount },
                txns.averageOf { it.amount }
            )
        }

    println("交易统计（大于100元）:")
    transactions.forEach { (category, stats) ->
        val (count, total, average) = stats
        println("  $category: $count笔, 总计¥$total, 平均¥$average")
    }
}
```

---

## 5.4 扩展函数与标准函数

### 5.4.1 扩展函数深入

```kotlin
// 字符串扩展函数
fun StringExtensionsDemo() {
    println("=== 字符串扩展函数 ===")

    // 基础字符串扩展
    fun String.isEmail(): Boolean {
        return this.matches(Regex("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"))
    }

    fun String.isPhoneNumber(): Boolean {
        return this.matches(Regex("^1[3-9]\\d{9}$"))
    }

    fun String.toCamelCase(): String {
        return this.split("_", "-")
            .mapIndexed { index, word ->
                if (index == 0) word.lowercase()
                else word.replaceFirstChar { it.uppercase() }
            }
            .joinToString("")
    }

    fun String.truncate(maxLength: Int, suffix: String = "..."): String {
        return if (this.length <= maxLength) this
        else this.take(maxLength - suffix.length) + suffix
    }

    fun String.removeHtmlTags(): String {
        return this.replace(Regex("<[^>]*>"), "")
    }

    // 使用扩展函数
    val email = "user@example.com"
    println("$email 是邮箱: ${email.isEmail()}")

    val phone = "13800138000"
    println("$phone 是手机号: ${phone.isPhoneNumber()}")

    val snakeCase = "hello_world_kotlin"
    println("$snakeCase 转驼峰: ${snakeCase.toCamelCase()}")

    val longText = "这是一段很长的文本，需要截断处理"
    println("截断前: $longText")
    println("截断后: ${longText.truncate(10)}")

    val htmlText = "<p>Hello <b>World</b></p>"
    println("HTML: $htmlText")
    println("去除标签: ${htmlText.removeHtmlTags()}")
}

// 集合扩展函数
fun collectionExtensionsDemo() {
    println("\n=== 集合扩展函数 ===")

    // 统计扩展
    fun <T> List<T>.frequency(): Map<T, Int> {
        return this.groupBy { it }.mapValues { it.value.size }
    }

    fun <T> List<T>.findDuplicates(): List<T> {
        return this.groupBy { it }
            .filter { it.value.size > 1 }
            .keys
            .toList()
    }

    fun <T> List<T>.takeRandom(n: Int): List<T> {
        return this.shuffled().take(n)
    }

    fun <T> List<T>.chunkBySize(chunkSize: Int): List<List<T>> {
        return this.windowed(chunkSize, chunkSize, true)
    }

    // 数值扩展
    fun List<Int>.median(): Double {
        val sorted = this.sorted()
        val size = sorted.size
        return if (size % 2 == 0) {
            (sorted[size / 2 - 1] + sorted[size / 2]) / 2.0
        } else {
            sorted[size / 2].toDouble()
        }
    }

    fun List<Double>.percentile(p: Double): Double {
        val sorted = this.sorted()
        val index = (p * (sorted.size - 1)).toInt()
        return sorted[index]
    }

    // 使用扩展函数
    val numbers = listOf(1, 2, 2, 3, 3, 3, 4, 5)
    println("原始列表: $numbers")
    println("频率统计: ${numbers.frequency()}")
    println("重复元素: ${numbers.findDuplicates()}")
    println("随机取3个: ${numbers.takeRandom(3)}")
    println("分块(大小2): ${numbers.chunkBySize(2)}")

    val scores = listOf(85, 92, 78, 96, 88, 73, 89)
    println("分数列表: $scores")
    println("中位数: ${scores.median()}")

    val values = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
    println("数值列表: $values")
    println("第90百分位: ${values.percentile(0.9)}")
}

// 类型扩展函数
fun typeExtensionsDemo() {
    println("\n=== 类型扩展函数 ===")

    // Int扩展
    fun Int.isEven(): Boolean = this % 2 == 0
    fun Int.isOdd(): Boolean = this % 2 == 1
    fun Int.isPrime(): Boolean {
        if (this <= 1) return false
        if (this <= 3) return true
        if (this % 2 == 0 || this % 3 == 0) return false

        var i = 5
        while (i * i <= this) {
            if (this % i == 0 || this % (i + 2) == 0) return false
            i += 6
        }
        return true
    }

    fun Int.toBinary(): String = Integer.toBinaryString(this)
    fun Int.toHex(): String = Integer.toHexString(this).uppercase()
    fun Int.formatAsPrice(): String = "¥${String.format("%,.2f", this.toDouble())}"

    // Boolean扩展
    fun Boolean.toInt(): Int = if (this) 1 else 0
    fun Boolean.toYesNo(): String = if (this) "是" else "否"
    fun Boolean.toEnableDisable(): String = if (this) "启用" else "禁用"

    // 使用扩展函数
    val number = 17
    println("$number 是偶数: ${number.isEven()}")
    println("$number 是奇数: ${number.isOdd()}")
    println("$number 是质数: ${number.isPrime()}")
    println("$number 二进制: ${number.toBinary()}")
    println("$number 十六进制: ${number.toHex()}")
    println("$number 价格格式: ${number.formatAsPrice()}")

    val isEnabled = true
    println("$isEnabled 转整数: ${isEnabled.toInt()}")
    println("$isEnabled 转是否: ${isEnabled.toYesNo()}")
    println("$isEnabled 转启用状态: ${isEnabled.toEnableDisable()}")
}
```

### 5.2.2 标准函数详解

```kotlin
// 标准函数的使用对比
fun standardFunctionsDemo() {
    println("=== 标准函数对比 ===")

    data class Person(var name: String, var age: Int, var email: String?)

    val person = Person("张三", 25, null)

    // let函数 - 对象转换和空安全处理
    val upperName = person.name.let {
        it.uppercase()
    }
    println("let转换: $upperName")

    val emailLength = person.email?.let { email ->
        if (email.isNotBlank()) email.length else 0
    } ?: 0
    println("邮箱长度: $emailLength")

    // also函数 - 配置对象（副作用）
    val configuredPerson = person.also {
        println("配置前: $it")
        it.age = 26
        if (it.email == null) {
            it.email = "zhangsan@example.com"
        }
        println("配置后: $it")
    }

    // apply函数 - 对象配置并返回对象本身
    val appliedPerson = person.apply {
        name = "李四"
        age = 30
        email = "lisi@example.com"
    }
    println("apply配置后: $appliedPerson")

    // run函数 - 对象配置和计算结果
    val personInfo = person.run {
        "$name ($age岁) - ${email ?: "无邮箱"}"
    }
    println("run计算结果: $personInfo")

    // with函数 - 对多个操作进行分组
    val personReport = with(person) {
        val header = "=== 用户报告 ==="
        val basic = "姓名: $name, 年龄: $age"
        val contact = "邮箱: ${email ?: "未设置"}"
        val status = when {
            age < 18 -> "未成年"
            age < 65 -> "成年"
            else -> "老年"
        }

        "$header\n$basic\n$contact\n状态: $status"
    }
    println("with报告:\n$personReport")
}

// 标准函数的实际应用场景
fun practicalStandardFunctionsDemo() {
    println("\n=== 标准函数实际应用 ===")

    // 场景1：数据库操作
    class DatabaseConnection {
        fun connect(): Boolean = true
        fun execute(sql: String): Int = 1
        fun close() = println("数据库连接已关闭")
    }

    // 使用let进行资源管理
    fun executeQuery(sql: String): Int? {
        val connection = DatabaseConnection()
        return if (connection.connect()) {
            connection.let { conn ->
                try {
                    conn.execute(sql)
                } finally {
                    conn.close()
                }
            }
        } else null
    }

    val result = executeQuery("SELECT * FROM users")
    println("查询结果: $result")

    // 场景2：对象构建
    class UserBuilder {
        var name: String = ""
        var age: Int = 0
        var email: String = ""

        fun build(): User {
            return User(name, age, email)
        }
    }

    data class User(val name: String, val age: Int, val email: String)

    // 使用apply构建对象
    val user = UserBuilder().apply {
        name = "王五"
        age = 28
        email = "wangwu@example.com"
    }.build()

    println("构建的用户: $user")

    // 场景3：链式调用
    data class Request(var url: String = "", var method: String = "GET",
                     var headers: MutableMap<String, String> = mutableMapOf(),
                     var body: String = "") {
        fun addHeader(key: String, value: String) = apply {
            headers[key] = value
        }

        fun setBody(body: String) = apply {
            this.body = body
        }

        fun setMethod(method: String) = apply {
            this.method = method
        }

        fun send(): String = "$method $url ${headers} $body"
    }

    val response = Request()
        .setUrl("https://api.example.com/users")
        .setMethod("POST")
        .addHeader("Content-Type", "application/json")
        .addHeader("Authorization", "Bearer token")
        .setBody("""{"name":"测试","age":25}""")
        .send()

    println("请求响应: $response")

    // 场景4：配置和验证
    class Config(var host: String = "", var port: Int = 0, var timeout: Int = 0) {
        fun validate(): Boolean {
            return host.isNotBlank() && port in 1..65535 && timeout > 0
        }

        fun toConnectionString(): String {
            return "$host:$port"
        }
    }

    fun createConfig(): Config? {
        return Config().apply {
            host = "localhost"
            port = 8080
            timeout = 5000
        }.takeIf { it.validate() }?.also {
            println("配置创建成功: ${it.toConnectionString()}")
        }
    }

    val config = createConfig()
    println("最终配置: $config")

    // 场景5：复杂计算
    data class Order(val items: List<OrderItem>, val discount: Double = 0.0) {
        val subtotal: Double get() = items.sumOf { it.price * it.quantity }
        val taxAmount: Double get() = subtotal * 0.08
        val discountAmount: Double get() = subtotal * discount
        val total: Double get() = subtotal + taxAmount - discountAmount
    }

    data class OrderItem(val name: String, val price: Double, val quantity: Int)

    val order = Order(
        items = listOf(
            OrderItem("商品A", 100.0, 2),
            OrderItem("商品B", 50.0, 3),
            OrderItem("商品C", 75.0, 1)
        ),
        discount = 0.1
    )

    val orderSummary = with(order) {
        buildString {
            appendLine("=== 订单摘要 ===")
            appendLine("小计: ¥${subtotal.format(2)}")
            appendLine("税额: ¥${taxAmount.format(2)}")
            appendLine("折扣: ¥${discountAmount.format(2)}")
            appendLine("总计: ¥${total.format(2)}")
        }
    }

    println(orderSummary)
}

// 数字格式化扩展函数
fun Double.format(digits: Int): String = String.format("%.${digits}f", this)
```

---

## 5.5 时间与日期处理

### 5.5.1 kotlinx-datetime库使用

```kotlin
// 依赖添加：implementation "org.jetbrains.kotlinx:kotlinx-datetime:0.4.0"

// 时间基础操作
fun dateTimeBasicsDemo() {
    println("=== 时间基础操作 ===")

    import kotlinx.datetime.*

    // 获取当前时间
    val now = Clock.System.now()
    println("当前时间: $now")

    // 转换为本地时间
    val timeZone = TimeZone.currentSystemDefault()
    val localDateTime = now.toLocalDateTime(timeZone)
    println("本地时间: $localDateTime")

    // 创建特定时间
    val specificDate = LocalDate(2024, 1, 15)
    val specificTime = LocalTime(14, 30, 0)
    val specificDateTime = LocalDateTime(specificDate, specificTime)
    println("特定日期时间: $specificDateTime")

    // 时间格式化
    val formatter = LocalDateTime.Format {
        year(); char('-'); monthNumber(); char('-'); dayOfMonth()
        char(' '); hour(); char(':'); minute(); char(':'); second()
    }
    val formattedTime = localDateTime.format(formatter)
    println("格式化时间: $formattedTime")

    // 时间解析
    val parsedDateTime = LocalDateTime.parse("2024-01-15 14:30:00", formatter)
    println("解析时间: $parsedDateTime")
}

// 时间计算操作
fun dateTimeCalculationsDemo() {
    println("\n=== 时间计算操作 ===")

    import kotlinx.datetime.*

    val now = Clock.System.now().toLocalDateTime(TimeZone.currentSystemDefault())

    // 日期加减
    val tomorrow = now.date.plus(1, DateTimeUnit.DAY)
    val nextWeek = now.date.plus(1, DateTimeUnit.WEEK)
    val nextMonth = now.date.plus(1, DateTimeUnit.MONTH)
    val nextYear = now.date.plus(1, DateTimeUnit.YEAR)

    println("今天: ${now.date}")
    println("明天: $tomorrow")
    println("下周: $nextWeek")
    println("下月: $nextMonth")
    println("明年: $nextYear")

    // 时间差计算
    val pastDate = LocalDate(2020, 1, 1)
    val dateDifference = now.date - pastDate
    println("从2020年1月1日到现在: ${dateDifference.days}天")

    // 工作日计算
    fun countWorkDays(startDate: LocalDate, endDate: LocalDate): Int {
        var count = 0
        var currentDate = startDate

        while (currentDate <= endDate) {
            if (currentDate.dayOfWeek != DayOfWeek.SATURDAY &&
                currentDate.dayOfWeek != DayOfWeek.SUNDAY) {
                count++
            }
            currentDate = currentDate.plus(1, DateTimeUnit.DAY)
        }
        return count
    }

    val startDate = LocalDate(2024, 1, 1)
    val endDate = LocalDate(2024, 1, 31)
    val workDays = countWorkDays(startDate, endDate)
    println("2024年1月工作日: $workDays天")

    // 时间段操作
    val startTime = LocalTime(9, 0)
    val endTime = LocalTime(17, 30)
    val workDuration = endTime - startTime
    println("工作时间: ${workDuration.toComponents { hours, minutes, _, _ ->
        "${hours}小时${minutes}分钟"
    }}")
}

// 实际应用：时间工具类
class DateTimeUtils {
    import kotlinx.datetime.*

    companion object {
        private val timeZone = TimeZone.currentSystemDefault()

        // 获取当前时间戳
        fun currentTimestamp(): Long = Clock.System.now().epochSeconds

        // 时间戳转日期时间
        fun timestampToDateTime(timestamp: Long): LocalDateTime {
            return Instant.fromEpochSeconds(timestamp).toLocalDateTime(timeZone)
        }

        // 日期时间转时间戳
        fun dateTimeToTimestamp(dateTime: LocalDateTime): Long {
            return dateTime.toInstant(timeZone).epochSeconds
        }

        // 格式化日期时间
        fun formatDateTime(dateTime: LocalDateTime, pattern: String = "yyyy-MM-dd HH:mm:ss"): String {
            return dateTime.format(
                LocalDateTime.Format {
                    when {
                        pattern.contains("yyyy") -> year()
                        pattern.contains("MM") -> monthNumber()
                        pattern.contains("dd") -> dayOfMonth()
                        pattern.contains("HH") -> hour()
                        pattern.contains("mm") -> minute()
                        pattern.contains("ss") -> second()
                    }
                }
            ).toString()
        }

        // 解析日期时间字符串
        fun parseDateTime(dateString: String, pattern: String = "yyyy-MM-dd HH:mm:ss"): LocalDateTime? {
            return try {
                LocalDateTime.parse(dateString)
            } catch (e: Exception) {
                null
            }
        }

        // 计算年龄
        fun calculateAge(birthDate: LocalDate): Int {
            val now = Clock.System.now().toLocalDateTime(timeZone).date
            return now.year - birthDate.year -
                   if (now.monthNumber < birthDate.monthNumber ||
                       (now.monthNumber == birthDate.monthNumber && now.dayOfMonth < birthDate.dayOfMonth)) 1 else 0
        }

        // 获取当月开始时间
        fun getMonthStart(year: Int, month: Int): LocalDateTime {
            return LocalDateTime(LocalDate(year, month, 1), LocalTime(0, 0, 0))
        }

        // 获取当月结束时间
        fun getMonthEnd(year: Int, month: Int): LocalDateTime {
            val lastDay = LocalDate(year, month, 1).let { date ->
                date.minus(1, DateTimeUnit.DAY)
            }
            return LocalDateTime(lastDay, LocalTime(23, 59, 59))
        }

        // 判断是否为工作日
        fun isWorkDay(date: LocalDate): Boolean {
            return date.dayOfWeek != DayOfWeek.SATURDAY &&
                   date.dayOfWeek != DayOfWeek.SUNDAY
        }

        // 获取下一个工作日
        fun getNextWorkDay(date: LocalDate): LocalDate {
            var nextDay = date.plus(1, DateTimeUnit.DAY)
            while (!isWorkDay(nextDay)) {
                nextDay = nextDay.plus(1, DateTimeUnit.DAY)
            }
            return nextDay
        }
    }
}

// 时间工具类的使用示例
fun dateTimeUtilsDemo() {
    println("\n=== DateTimeUtils使用示例 ===")

    // 当前时间操作
    val currentTimestamp = DateTimeUtils.currentTimestamp()
    println("当前时间戳: $currentTimestamp")

    val currentDateTime = DateTimeUtils.timestampToDateTime(currentTimestamp)
    println("当前日期时间: $currentDateTime")

    // 日期格式化和解析
    val formattedDate = DateTimeUtils.formatDateTime(currentDateTime)
    println("格式化日期: $formattedDate")

    val parsedDate = DateTimeUtils.parseDateTime("2024-01-15 14:30:00")
    println("解析日期: $parsedDate")

    // 年龄计算
    val birthDate = LocalDate(1990, 5, 15)
    val age = DateTimeUtils.calculateAge(birthDate)
    println("生日${birthDate}的年龄: $age岁")

    // 月度时间范围
    val monthStart = DateTimeUtils.getMonthStart(2024, 1)
    val monthEnd = DateTimeUtils.getMonthEnd(2024, 1)
    println("2024年1月开始: $monthStart")
    println("2024年1月结束: $monthEnd")

    // 工作日操作
    val testDate = LocalDate(2024, 1, 12) // 假设是星期五
    println("$testDate 是工作日: ${DateTimeUtils.isWorkDay(testDate)}")

    val nextWorkDay = DateTimeUtils.getNextWorkDay(testDate)
    println("下一个工作日: $nextWorkDay")
}
```

---

## 5.6 实战项目：数据处理工具集

### 5.6.1 数据分析工具

```kotlin
// 数据分析工具类
class DataAnalyzer {
    // 统计分析
    data class Statistics(
        val count: Int,
        val sum: Double,
        val average: Double,
        val min: Double,
        val max: Double,
        val median: Double,
        val variance: Double,
        val standardDeviation: Double
    )

    fun calculateStatistics(numbers: List<Double>): Statistics {
        if (numbers.isEmpty()) {
            return Statistics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        }

        val sortedNumbers = numbers.sorted()
        val count = numbers.size
        val sum = numbers.sum()
        val average = sum / count
        val min = numbers.minOrNull() ?: 0.0
        val max = numbers.maxOrNull() ?: 0.0

        // 中位数计算
        val median = if (count % 2 == 0) {
            (sortedNumbers[count / 2 - 1] + sortedNumbers[count / 2]) / 2.0
        } else {
            sortedNumbers[count / 2]
        }

        // 方差计算
        val variance = numbers.map { (it - average) * (it - average) }.average()
        val standardDeviation = Math.sqrt(variance)

        return Statistics(count, sum, average, min, max, median, variance, standardDeviation)
    }

    // 频率分析
    fun <T> frequencyAnalysis(items: List<T>): Map<T, Int> {
        return items.groupBy { it }.mapValues { it.value.size }
    }

    // 百分位数分析
    fun percentiles(numbers: List<Double>, percentiles: List<Double> = listOf(0.25, 0.5, 0.75, 0.9, 0.95, 0.99)): Map<Double, Double> {
        val sorted = numbers.sorted()
        return percentiles.associateWith { percentile ->
            val index = (percentile * (sorted.size - 1)).toInt()
            sorted[index]
        }
    }

    // 异常值检测（IQR方法）
    fun detectOutliers(numbers: List<Double>): List<Double> {
        val sorted = numbers.sorted()
        val q1Index = (sorted.size * 0.25).toInt()
        val q3Index = (sorted.size * 0.75).toInt()
        val q1 = sorted[q1Index]
        val q3 = sorted[q3Index]
        val iqr = q3 - q1

        val lowerBound = q1 - 1.5 * iqr
        val upperBound = q3 + 1.5 * iqr

        return numbers.filter { it < lowerBound || it > upperBound }
    }
}

// 数据清洗工具类
class DataCleaner {
    // 移除空值
    fun <T> removeNulls(list: List<T?>): List<T> {
        return list.filterNotNull()
    }

    // 移除重复项
    fun <T> removeDuplicates(list: List<T>): List<T> {
        return list.distinct()
    }

    // 移除异常值
    fun removeOutliers(numbers: List<Double>): List<Double> {
        val analyzer = DataAnalyzer()
        val outliers = analyzer.detectOutliers(numbers)
        return numbers.filter { it !in outliers }
    }

    // 填充缺失值
    fun fillMissingValues(numbers: List<Double?>, strategy: FillStrategy = FillStrategy.MEAN): List<Double> {
        val validNumbers = numbers.filterNotNull()
        if (validNumbers.isEmpty()) return numbers.map { 0.0 }

        val fillValue = when (strategy) {
            FillStrategy.MEAN -> validNumbers.average()
            FillStrategy.MEDIAN -> {
                val sorted = validNumbers.sorted()
                if (sorted.size % 2 == 0) {
                    (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2.0
                } else {
                    sorted[sorted.size / 2]
                }
            }
            FillStrategy.MODE -> {
                val frequency = validNumbers.groupBy { it }.mapValues { it.value.size }
                frequency.maxByOrNull { it.value }?.key ?: 0.0
            }
            FillStrategy.ZERO -> 0.0
        }

        return numbers.map { it ?: fillValue }
    }

    enum class FillStrategy {
        MEAN, MEDIAN, MODE, ZERO
    }
}

// 数据转换工具类
class DataTransformer {
    // 标准化
    fun normalize(numbers: List<Double>): List<Double> {
        val min = numbers.minOrNull() ?: 0.0
        val max = numbers.maxOrNull() ?: 0.0
        val range = max - min

        return if (range == 0.0) {
            List(numbers.size) { 0.0 }
        } else {
            numbers.map { (it - min) / range }
        }
    }

    // Z-score标准化
    fun standardize(numbers: List<Double>): List<Double> {
        val analyzer = DataAnalyzer()
        val stats = analyzer.calculateStatistics(numbers)

        return if (stats.standardDeviation == 0.0) {
            List(numbers.size) { 0.0 }
        } else {
            numbers.map { (it - stats.average) / stats.standardDeviation }
        }
    }

    // 对数变换
    fun logTransform(numbers: List<Double>): List<Double> {
        return numbers.map { Math.log10(it) }
    }

    // 分类数据编码
    fun encodeCategories(categories: List<String>): Map<String, List<Int>> {
        val uniqueCategories = categories.distinct()
        val encoding = uniqueCategories.mapIndexed { index, category ->
            category to List(uniqueCategories.size) { if (it == index) 1 else 0 }
        }.toMap()
        return encoding
    }
}
```

### 5.6.2 实际数据分析示例

```kotlin
// 销售数据分析
data class SalesRecord(
    val date: String,
    val product: String,
    val category: String,
    val region: String,
    val amount: Double,
    val quantity: Int
)

class SalesDataAnalyzer {
    private val analyzer = DataAnalyzer()
    private val cleaner = DataCleaner()
    private val transformer = DataTransformer()

    fun analyzeSalesData(records: List<SalesRecord>): SalesAnalysisResult {
        // 数据清洗
        val cleanedRecords = records.filter { it.amount > 0 && it.quantity > 0 }

        // 基本统计
        val amounts = cleanedRecords.map { it.amount }
        val statistics = analyzer.calculateStatistics(amounts)

        // 按产品分析
        val productAnalysis = cleanedRecords
            .groupBy { it.product }
            .mapValues { (_, records) ->
                ProductAnalysis(
                    totalSales = records.sumOf { it.amount },
                    totalQuantity = records.sumOf { it.quantity },
                    averagePrice = records.map { it.amount / it.quantity }.average(),
                    salesCount = records.size
                )
            }

        // 按分类分析
        val categoryAnalysis = cleanedRecords
            .groupBy { it.category }
            .mapValues { (_, records) ->
                CategoryAnalysis(
                    totalSales = records.sumOf { it.amount },
                    productCount = records.map { it.product }.distinct().size,
                    salesCount = records.size
                )
            }

        // 按地区分析
        val regionAnalysis = cleanedRecords
            .groupBy { it.region }
            .mapValues { (_, records) ->
                RegionAnalysis(
                    totalSales = records.sumOf { it.amount },
                    salesCount = records.size,
                    topProducts = records
                        .groupBy { it.product }
                        .mapValues { it.value.sumOf { it.amount } }
                        .toList()
                        .sortedByDescending { it.second }
                        .take(3)
                        .map { it.first }
                )
            }

        // 时间趋势分析
        val monthlyTrend = cleanedRecords
            .groupBy { it.date.substring(0, 7) } // YYYY-MM
            .mapValues { (_, records) ->
                records.sumOf { it.amount }
            }
            .toList()
            .sortedBy { it.first }

        // 异常销售检测
        val outliers = analyzer.detectOutliers(amounts)
        val outlierRecords = cleanedRecords.filter { it.amount in outliers }

        return SalesAnalysisResult(
            statistics = statistics,
            productAnalysis = productAnalysis,
            categoryAnalysis = categoryAnalysis,
            regionAnalysis = regionAnalysis,
            monthlyTrend = monthlyTrend,
            outliers = outlierRecords
        )
    }
}

data class SalesAnalysisResult(
    val statistics: DataAnalyzer.Statistics,
    val productAnalysis: Map<String, ProductAnalysis>,
    val categoryAnalysis: Map<String, CategoryAnalysis>,
    val regionAnalysis: Map<String, RegionAnalysis>,
    val monthlyTrend: List<Pair<String, Double>>,
    val outliers: List<SalesRecord>
)

data class ProductAnalysis(
    val totalSales: Double,
    val totalQuantity: Int,
    val averagePrice: Double,
    val salesCount: Int
)

data class CategoryAnalysis(
    val totalSales: Double,
    val productCount: Int,
    val salesCount: Int
)

data class RegionAnalysis(
    val totalSales: Double,
    val salesCount: Int,
    val topProducts: List<String>
)

// 销售数据分析示例
fun salesAnalysisDemo() {
    println("=== 销售数据分析示例 ===")

    // 模拟销售数据
    val salesData = listOf(
        SalesRecord("2024-01-15", "Laptop", "Electronics", "北京", 1200.0, 5),
        SalesRecord("2024-01-16", "Mouse", "Electronics", "上海", 50.0, 20),
        SalesRecord("2024-01-17", "Keyboard", "Electronics", "广州", 80.0, 15),
        SalesRecord("2024-02-10", "Laptop", "Electronics", "深圳", 1300.0, 3),
        SalesRecord("2024-02-15", "Monitor", "Electronics", "北京", 300.0, 8),
        SalesRecord("2024-02-20", "Mouse", "Electronics", "上海", 45.0, 25),
        SalesRecord("2024-03-05", "Laptop", "Electronics", "成都", 1250.0, 4),
        SalesRecord("2024-03-10", "Keyboard", "Electronics", "杭州", 75.0, 18),
        SalesRecord("2024-03-15", "Monitor", "Electronics", "北京", 320.0, 6),
        SalesRecord("2024-03-20", "Mouse", "Electronics", "深圳", 55.0, 12)
    )

    val analyzer = SalesDataAnalyzer()
    val result = analyzer.analyzeSalesData(salesData)

    // 打印分析结果
    println("=== 销售统计 ===")
    val stats = result.statistics
    println("总销售额: ¥${stats.sum.format(2)}")
    println("平均销售额: ¥${stats.average.format(2)}")
    println("最大销售额: ¥${stats.max.format(2)}")
    println("最小销售额: ¥${stats.min.format(2)}")
    println("中位数: ¥${stats.median.format(2)}")

    println("\n=== 产品分析 ===")
    result.productAnalysis.entries
        .sortedByDescending { it.value.totalSales }
        .forEach { (product, analysis) ->
            println("$product: 总销售额¥${analysis.totalSales.format(2)}, " +
                    "总数量${analysis.totalQuantity}, " +
                    "均价¥${analysis.averagePrice.format(2)}")
        }

    println("\n=== 地区分析 ===")
    result.regionAnalysis.forEach { (region, analysis) ->
        println("$region: 总销售额¥${analysis.totalSales.format(2)}, " +
                "销售笔数${analysis.salesCount}, " +
                "热销产品: ${analysis.topProducts.joinToString(", ")}")
    }

    println("\n=== 月度趋势 ===")
    result.monthlyTrend.forEach { (month, sales) ->
        println("$month: ¥${sales.format(2)}")
    }

    if (result.outliers.isNotEmpty()) {
        println("\n=== 异常销售 ===")
        result.outliers.forEach { record ->
            println("${record.date} - ${record.product}: ¥${record.amount.format(2)} " +
                    "(数量: ${record.quantity})")
        }
    }
}

fun Double.format(digits: Int): String = String.format("%.${digits}f", this)
```

---

## 5.7 本章小结

### ✅ 核心概念掌握

通过本章学习，您已经掌握了Kotlin标准库和集合框架的强大功能：

1. **标准库概览**
   - 标准库的结构和组成
   - 常用工具类和函数
   - 数学运算和随机数生成
   - 类型转换和条件操作

2. **集合类型详解**
   - List的创建、操作和高级特性
   - Set的特性和集合运算
   - Map的键值对操作和实际应用
   - 集合的性能优化技巧

3. **范围与序列**
   - Range的使用和高级操作
   - Sequence的惰性计算特性
   - 性能优化和内存效率
   - 实际业务场景应用

4. **扩展函数与标准函数**
   - 扩展函数的定义和使用
   - 标准函数的选择和应用场景
   - 链式调用和函数式编程
   - 代码复用和最佳实践

5. **时间与日期处理**
   - kotlinx-datetime库的使用
   - 时间计算和格式化
   - 实际应用中的时间处理
   - 工具类的设计和实现

### ✅ 相比Java集合框架的优势

| 特性 | Java Collections | Kotlin Collections | 优势程度 |
|------|------------------|-------------------|----------|
| 不可变性 | 默认可变 | 默认不可变 | ⭐⭐⭐⭐⭐ |
| API丰富度 | 基础操作 | 丰富的函数式API | ⭐⭐⭐⭐⭐ |
| 空安全 | 需要手动检查 | 内置空安全 | ⭐⭐⭐⭐⭐ |
| 扩展性 | 工具类模式 | 扩展函数 | ⭐⭐⭐⭐⭐ |
| 代码简洁性 | 冗长的循环和条件 | 简洁的函数式代码 | ⭐⭐⭐⭐ |
| 类型推断 | 有限推断 | 强大类型推断 | ⭐⭐⭐⭐ |
| 标准函数 | 无 | let/run/with/apply/also | ⭐⭐⭐⭐⭐ |

### ✅ 实战要点

1. **集合选择原则**
   - 优先使用不可变集合
   - 根据访问模式选择合适的集合类型
   - 合理使用Sequence优化性能
   - 注意集合操作的内存消耗

2. **性能优化建议**
   - 大数据集使用Sequence
   - 避免不必要的集合转换
   - 合理使用缓存和延迟计算
   - 选择合适的数据结构

3. **代码质量**
   - 充分利用扩展函数提高复用性
   - 选择合适的标准函数
   - 保持代码的可读性和维护性
   - 遵循Kotlin的编码规范

### 📚 下一步学习

下一章我们将探索**Kotlin与Java互操作**，包括：
- Kotlin调用Java代码的最佳实践
- Java调用Kotlin代码的注意事项
- 注解处理兼容性
- SAM转换与函数式接口
- 项目迁移策略

这将帮助您在现有Java项目中顺利引入Kotlin！

---

## 📝 章节练习

### 基础练习
1. 实现一个数据统计工具：
   - 计算平均值、中位数、众数
   - 实现数据标准化
   - 支持异常值检测
   - 提供数据可视化功能

2. 重构以下Java代码为Kotlin集合操作：
```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
List<Integer> result = new ArrayList<>();
for (Integer num : numbers) {
    if (num % 2 == 0 && num > 4) {
        result.add(num * num);
    }
}
Collections.sort(result);
```

### 进阶练习
1. 创建一个时间工具库：
   - 支持多种时间格式
   - 实现时间计算功能
   - 支持时区转换
   - 提供工作日计算

2. 实现一个集合分析器：
   - 支持多种数据类型分析
   - 提供频率分析功能
   - 实现数据清洗和转换
   - 支持自定义分析规则

### 挑战练习
1. 构建一个数据处理管道：
   - 支持多种数据源
   - 实现可配置的处理步骤
   - 支持并行处理
   - 提供监控和日志功能

2. 设计一个通用集合工具框架：
   - 支持插件式扩展
   - 实现类型安全操作
   - 提供性能优化
   - 支持自定义集合类型

---

**恭喜完成Kotlin标准库与集合框架的学习！您现在已经掌握了Kotlin丰富的标准工具库，可以高效地处理各种数据操作任务了！**