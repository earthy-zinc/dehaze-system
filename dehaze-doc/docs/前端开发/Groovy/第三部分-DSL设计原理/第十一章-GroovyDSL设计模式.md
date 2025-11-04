# 第十一章：Groovy DSL设计模式

> 设计模式是解决特定问题的成熟方案。在DSL开发中，合理运用设计模式能够让DSL更加优雅、易用和可维护。本章将深入探讨Groovy DSL开发中的常用设计模式。

## 11.1 Builder模式

### 11.1.1 经典Builder模式

Builder模式是DSL开发中最常用的模式之一，它允许用户通过链式调用逐步构建复杂对象。

```groovy
// 经典Builder模式示例
class EmailBuilder {
    private String from
    private String to
    private String subject
    private String body
    private List<String> attachments = []
    private Map<String, String> headers = [:]

    def from(String email) {
        this.from = email
        return this
    }

    def to(String email) {
        this.to = email
        return this
    }

    def subject(String subject) {
        this.subject = subject
        return this
    }

    def body(String body) {
        this.body = body
        return this
    }

    def attach(String filename) {
        attachments.add(filename)
        return this
    }

    def header(String name, String value) {
        headers[name] = value
        return this
    }

    Email build() {
        validate()
        return new Email(from, to, subject, body, attachments, headers)
    }

    private void validate() {
        if (!from) throw new IllegalStateException("From address is required")
        if (!to) throw new IllegalStateException("To address is required")
        if (!subject) throw new IllegalStateException("Subject is required")
    }
}

class Email {
    String from
    String to
    String subject
    String body
    List<String> attachments
    Map<String, String> headers

    Email(from, to, subject, body, attachments, headers) {
        this.from = from
        this.to = to
        this.subject = subject
        this.body = body
        this.attachments = attachments
        this.headers = headers
    }

    String toString() {
        "Email(from=${from}, to=${to}, subject=${subject})"
    }
}

// 使用Builder模式
def email = new EmailBuilder()
    .from("sender@example.com")
    .to("receiver@example.com")
    .subject("Hello")
    .body("This is the email body")
    .attach("document.pdf")
    .attach("image.png")
    .header("X-Priority", "High")
    .build()

println email
```

### 11.1.2 Groovy增强的Builder模式

利用Groovy的闭包和元编程特性，可以让Builder模式更加优雅。

```groovy
// Groovy增强的Builder模式
class EnhancedEmailBuilder {
    private Email email = new Email()

    def from(String email) {
        this.email.from = email
        return this
    }

    def to(String email) {
        this.email.to = email
        return this
    }

    def subject(String subject) {
        this.email.subject = subject
        return this
    }

    def body(String body) {
        this.email.body = body
        return this
    }

    // 使用闭包进行嵌套配置
    def attachments(Closure closure) {
        def builder = new AttachmentBuilder()
        closure.delegate = builder
        closure.resolveStrategy = Closure.DELEGATE_FIRST
        closure()
        email.attachments = builder.attachments
        return this
    }

    def headers(Closure closure) {
        def builder = new HeaderBuilder()
        closure.delegate = builder
        closure.resolveStrategy = Closure.DELEGATE_FIRST
        closure()
        email.headers = builder.headers
        return this
    }

    def build() {
        email.validate()
        return email
    }

    // 使用methodMissing实现动态属性设置
    def methodMissing(String name, args) {
        if (name.startsWith('with')) {
            def propertyName = name.substring(4).uncapitalize()
            if (email.metaClass.hasProperty(email, propertyName)) {
                email."${propertyName}" = args[0]
                return this
            }
        }
        throw new MissingMethodException(name, this.class, args)
    }
}

class AttachmentBuilder {
    List<String> attachments = []

    def file(String filename) {
        attachments.add(filename)
        return this
    }

    def methodMissing(String name, args) {
        // 支持动态文件类型
        if (args && args[0] instanceof String) {
            attachments.add(args[0])
            return this
        }
        throw new MissingMethodException(name, this.class, args)
    }
}

class HeaderBuilder {
    Map<String, String> headers = [:]

    def methodMissing(String name, args) {
        if (args && args[0] instanceof String) {
            headers[name] = args[0]
            return this
        }
        throw new MissingMethodException(name, this.class, args)
    }
}

// 使用增强的Builder
def enhancedEmail = new EnhancedEmailBuilder()
    .from("sender@example.com")
    .to("receiver@example.com")
    .withSubject("Hello World")
    .withBody("Enhanced email content")
    .attachments {
        file "document.pdf"
        file "presentation.pptx"
        image "logo.png"
    }
    .headers {
        X_Priority "High"
        X_Mailer "GroovyMailer"
    }
    .build()

println enhancedEmail
```

### 11.1.3 领域特定Builder示例

```groovy
// HTTP请求Builder示例
class HttpRequestBuilder {
    private String method = "GET"
    private String url
    private Map<String, String> headers = [:]
    private Map<String, Object> queryParams = [:]
    private Object body

    def get(String url) {
        this.method = "GET"
        this.url = url
        return this
    }

    def post(String url) {
        this.method = "POST"
        this.url = url
        return this
    }

    def put(String url) {
        this.method = "PUT"
        this.url = url
        return this
    }

    def delete(String url) {
        this.method = "DELETE"
        this.url = url
        return this
    }

    def header(String name, String value) {
        headers[name] = value
        return this
    }

    def query(String name, Object value) {
        queryParams[name] = value
        return this
    }

    def json(Object data) {
        body = data
        header("Content-Type", "application/json")
        return this
    }

    def form(Map<String, String> data) {
        body = data
        header("Content-Type", "application/x-www-form-urlencoded")
        return this
    }

    def auth(String token) {
        header("Authorization", "Bearer ${token}")
        return this
    }

    def execute(Closure callback = null) {
        def request = new HttpRequest(method, url, headers, queryParams, body)
        println "Executing HTTP ${method} ${url}"

        if (callback) {
            callback.delegate = this
            callback(request)
        }

        return request.send()
    }

    def onSuccess(Closure callback) {
        this.onSuccess = callback
        return this
    }

    def onError(Closure callback) {
        this.onError = callback
        return this
    }
}

// 使用HTTP Builder
def response = new HttpRequestBuilder()
    .get("https://api.example.com/users")
    .query("page", 1)
    .query("limit", 20)
    .header("Accept", "application/json")
    .auth("your-api-token")
    .execute { request ->
        println "Request: ${request}"
    }

// JSON POST请求示例
def postResponse = new HttpRequestBuilder()
    .post("https://api.example.com/users")
    .json([
        name: "John Doe",
        email: "john@example.com",
        age: 30
    ])
    .header("X-API-Key", "your-api-key")
    .onSuccess { response -> println "Success: ${response}" }
    .onError { error -> println "Error: ${error}" }
    .execute()
```

## 11.2 Command Chain模式

### 11.2.1 方法链调用模式

Command Chain模式允许通过连续的方法调用来构建复杂的行为。

```groovy
// Command Chain模式示例
class QueryBuilder {
    private String selectClause = "*"
    private String fromClause
    private List<String> whereConditions = []
    private List<String> orderByFields = []
    private Integer limitValue
    private Integer offsetValue

    def select(String... fields) {
        if (fields) {
            selectClause = fields.join(", ")
        }
        return this
    }

    def from(String table) {
        fromClause = table
        return this
    }

    def where(String condition) {
        whereConditions.add(condition)
        return this
    }

    def and(String condition) {
        whereConditions.add("AND ${condition}")
        return this
    }

    def or(String condition) {
        whereConditions.add("OR ${condition}")
        return this
    }

    def orderBy(String field, String direction = "ASC") {
        orderByFields.add("${field} ${direction}")
        return this
    }

    def limit(int count) {
        limitValue = count
        return this
    }

    def offset(int count) {
        offsetValue = count
        return this
    }

    String build() {
        def sql = "SELECT ${selectClause} FROM ${fromClause}"

        if (whereConditions) {
            sql += " WHERE " + whereConditions.join(" ")
        }

        if (orderByFields) {
            sql += " ORDER BY " + orderByFields.join(", ")
        }

        if (limitValue) {
            sql += " LIMIT ${limitValue}"
        }

        if (offsetValue) {
            sql += " OFFSET ${offsetValue}"
        }

        return sql
    }

    // 支持Groovy的方法链特性
    def methodMissing(String name, args) {
        if (name.startsWith('findBy')) {
            def field = name.substring(6).uncapitalize()
            where("${field} = ?", args[0])
            return this
        }
        throw new MissingMethodException(name, this.class, args)
    }
}

// 使用Query Builder
def query = new QueryBuilder()
    .select("name", "email", "age")
    .from("users")
    .where("age > 18")
    .and("status = 'active'")
    .orderBy("name")
    .limit(10)

println query.build()

// 支持动态方法调用
def dynamicQuery = new QueryBuilder()
    .select("name", "email")
    .from("users")
    .findByEmail("john@example.com")  // 动态方法
    .and("age > 25")

println dynamicQuery.build()
```

### 11.2.2 流畅接口模式

流畅接口（Fluent Interface）让方法调用读起来像自然语言。

```groovy
// 流畅接口示例：数据验证DSL
class ValidationChain {
    private Object target
    private List<ValidationRule> rules = []
    private List<String> errors = []

    ValidationChain(Object target) {
        this.target = target
    }

    def must(Closure condition, String message) {
        rules.add(new ValidationRule(condition, message))
        return this
    }

    def notNull(String message = "Value cannot be null") {
        must({ it != null }, message)
        return this
    }

    def notEmpty(String message = "Value cannot be empty") {
        must({ it != null && !it.toString().trim().isEmpty() }, message)
        return this
    }

    def minLength(int length, String message = null) {
        def defaultMessage = "Value must be at least ${length} characters"
        must({ it != null && it.toString().length() >= length }, message ?: defaultMessage)
        return this
    }

    def maxLength(int length, String message = null) {
        def defaultMessage = "Value must not exceed ${length} characters"
        must({ it != null && it.toString().length() <= length }, message ?: defaultMessage)
        return this
    }

    def matches(String pattern, String message = null) {
        def defaultMessage = "Value does not match required pattern"
        must({ it != null && it.toString() ==~ pattern }, message ?: defaultMessage)
        return this
    }

    def email(String message = "Must be a valid email address") {
        def pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
        matches(pattern, message)
        return this
    }

    def range(Number min, Number max, String message = null) {
        def defaultMessage = "Value must be between ${min} and ${max}"
        must({ it != null && it instanceof Number && it >= min && it <= max },
             message ?: defaultMessage)
        return this
    }

    def custom(Closure validator, String message) {
        must(validator, message)
        return this
    }

    def validate() {
        errors.clear()
        rules.each { rule ->
            if (!rule.condition(target)) {
                errors.add(rule.message)
            }
        }
        return errors.isEmpty()
    }

    def getErrors() {
        return errors
    }

    def throwIfInvalid() {
        if (!validate()) {
            throw new ValidationException("Validation failed: ${errors.join(', ')}")
        }
        return this
    }

    // 支持 && 语法
    def and(Closure additionalValidation) {
        additionalValidation.delegate = this
        additionalValidation.resolveStrategy = Closure.DELEGATE_FIRST
        additionalValidation(target)
        return this
    }
}

class ValidationRule {
    Closure condition
    String message

    ValidationRule(Closure condition, String message) {
        this.condition = condition
        this.message = message
    }
}

class ValidationException extends RuntimeException {
    ValidationException(String message) {
        super(message)
    }
}

// 使用流畅接口进行验证
class User {
    String name
    String email
    int age
}

def user = new User(name: "", email: "invalid-email", age: 15)

def validation = new ValidationChain(user)
    .notNull("User object is required")
    .and {
        it.notNull("Name is required")
            .notEmpty("Name cannot be empty")
            .minLength(2, "Name must be at least 2 characters")
            .maxLength(50, "Name cannot exceed 50 characters")
    }
    .and {
        it.email("Email must be valid")
    }
    .and {
        it.range(18, 120, "Age must be between 18 and 120")
    }

if (!validation.validate()) {
    println "Validation errors:"
    validation.errors.each { error ->
        println "- ${error}"
    }
}

// 简化的使用方式
try {
    new ValidationChain("test@example.com")
        .notEmpty("Email is required")
        .email()
        .throwIfInvalid()

    println "Email validation passed"
} catch (ValidationException e) {
    println "Email validation failed: ${e.message}"
}
```

## 11.3 嵌套闭包模式

### 11.3.1 分层配置模式

嵌套闭包模式允许创建层次化的配置结构，非常适合复杂的配置需求。

```groovy
// 嵌套闭包模式：应用配置DSL
class ApplicationConfig {
    String name
    String version
    ServerConfig server = new ServerConfig()
    DatabaseConfig database = new DatabaseConfig()
    List<ModuleConfig> modules = []

    void name(String name) { this.name = name }
    void version(String version) { this.version = version }

    def server(Closure closure) {
        closure.delegate = server
        closure.resolveStrategy = Closure.DELEGATE_FIRST
        closure()
    }

    def database(Closure closure) {
        closure.delegate = database
        closure.resolveStrategy = Closure.DELEGATE_FIRST
        closure()
    }

    def module(Closure closure) {
        def module = new ModuleConfig()
        closure.delegate = module
        closure.resolveStrategy = Closure.DELEGATE_FIRST
        closure()
        modules.add(module)
    }

    String toString() {
        """Application: ${name} ${version}
        Server: ${server}
        Database: ${database}
        Modules: ${modules.collect { it.name }.join(', ')}"""
    }
}

class ServerConfig {
    String host = "localhost"
    int port = 8080
    String protocol = "HTTP"
    boolean sslEnabled = false
    Map<String, String> headers = [:]

    void host(String host) { this.host = host }
    void port(int port) { this.port = port }
    void protocol(String protocol) { this.protocol = protocol }
    void ssl(boolean enabled) { this.sslEnabled = enabled }

    def headers(Closure closure) {
        def builder = new MapBuilder(headers)
        closure.delegate = builder
        closure.resolveStrategy = Closure.DELEGATE_FIRST
        closure()
    }

    String toString() {
        "${protocol}://${host}:${port} (SSL: ${sslEnabled})"
    }
}

class DatabaseConfig {
    String url
    String username
    String password
    Map<String, String> properties = [:]
    PoolConfig pool = new PoolConfig()

    void url(String url) { this.url = url }
    void username(String username) { this.username = username }
    void password(String password) { this.password = password }

    def properties(Closure closure) {
        def builder = new MapBuilder(properties)
        closure.delegate = builder
        closure.resolveStrategy = Closure.DELEGATE_FIRST
        closure()
    }

    def pool(Closure closure) {
        closure.delegate = pool
        closure.resolveStrategy = Closure.DELEGATE_FIRST
        closure()
    }

    String toString() {
        "Database: ${url} (User: ${username})"
    }
}

class PoolConfig {
    int maxConnections = 10
    int minConnections = 1
    int timeout = 30000

    void maxConnections(int count) { this.maxConnections = count }
    void minConnections(int count) { this.minConnections = count }
    void timeout(int ms) { this.timeout = ms }
}

class ModuleConfig {
    String name
    String version
    boolean enabled = true
    Map<String, Object> settings = [:]

    void name(String name) { this.name = name }
    void version(String version) { this.version = version }
    void enabled(boolean enabled) { this.enabled = enabled }

    def settings(Closure closure) {
        def builder = new MapBuilder(settings)
        closure.delegate = builder
        closure.resolveStrategy = Closure.DELEGATE_FIRST
        closure()
    }
}

class MapBuilder {
    private Map target

    MapBuilder(Map target) {
        this.target = target
    }

    def methodMissing(String name, args) {
        if (args && args[0] != null) {
            target[name] = args[0]
        }
    }
}

// 使用嵌套闭包模式
def config = new ApplicationConfig()

config.name "My Application"
config.version "1.0.0"

config.server {
    host "api.example.com"
    port 443
    protocol "HTTPS"
    ssl true
    headers {
        "X-API-Version" "1.0"
        "X-Client-ID" "myapp"
    }
}

config.database {
    url "jdbc:mysql://prod-db:3306/myapp"
    username "appuser"
    password "apppassword"
    properties {
        "useSSL" true
        "verifyServerCertificate" false
        "autoReconnect" true
    }
    pool {
        maxConnections 50
        minConnections 5
        timeout 60000
    }
}

config.module {
    name "user-management"
    version "2.1.0"
    enabled true
    settings {
        "enableRegistration" true
        "passwordMinLength" 8
        "maxLoginAttempts" 5
    }
}

config.module {
    name "notification"
    version "1.5.2"
    enabled true
    settings {
        "emailEnabled" true
        "smsEnabled" false
        "pushEnabled" true
    }
}

println config
```

### 11.3.2 动态嵌套配置

```groovy
// 动态嵌套配置：支持任意深度的配置
class DynamicConfigBuilder {
    private Map config = [:]

    def methodMissing(String name, args) {
        if (args && args[0] instanceof Closure) {
            def nestedBuilder = new DynamicConfigBuilder()
            args[0].delegate = nestedBuilder
            args[0].resolveStrategy = Closure.DELEGATE_FIRST
            args[0]()
            config[name] = nestedBuilder.config
            return this
        } else if (args) {
            config[name] = args[0]
            return this
        }
        return config[name]
    }

    def propertyMissing(String name) {
        return config[name]
    }

    def propertyMissing(String name, value) {
        config[name] = value
    }

    Map build() {
        return config
    }

    String toString() {
        def pretty = { Map map, String indent = "" ->
            map.collectMany { key, value ->
                if (value instanceof Map) {
                    ["${indent}${key}:", pretty(value, indent + "  ")].flatten()
                } else {
                    ["${indent}${key}: ${value}"]
                }
            }
        }

        pretty(config).join("\n")
    }
}

// 使用动态嵌套配置
def dynamicConfig = new DynamicConfigBuilder()

dynamicConfig {
    application {
        name "Dynamic App"
        version "1.0.0"
        environment "production"
    }

    server {
        web {
            port 8080
            threads 50
            timeout 30000
        }

        database {
            primary {
                host "db1.example.com"
                port 5432
                name "appdb"
            }

            replica {
                host "db2.example.com"
                port 5432
                name "appdb_replica"
            }
        }
    }

    features {
        authentication {
            enabled true
            providers ["local", "oauth", "sso"]

            oauth {
                google {
                    clientId "google-client-id"
                    clientSecret "google-secret"
                }

                github {
                    clientId "github-client-id"
                    clientSecret "github-secret"
                }
            }
        }

        caching {
            type "redis"
            ttl 3600
            maxSize "100MB"
        }
    }
}

println "Dynamic Configuration:"
println dynamicConfig
```

## 11.4 扩展方法模式

### 11.1.1 MetaClass扩展

使用Groovy的元编程能力，可以为现有类添加DSL方法。

```groovy
// 扩展方法模式：为现有类添加DSL方法
class DSLEnhancements {
    static void enhance() {
        // 为String添加文件操作DSL
        String.metaClass.asFile = { -> new File(delegate) }
        String.metaClass.readAsText = { encoding = "UTF-8" -> delegate.asFile().getText(encoding) }
        String.metaClass.writeAsText = { content, encoding = "UTF-8" -> delegate.asFile().setText(content, encoding) }
        String.metaClass.appendAsText = { content, encoding = "UTF-8" -> delegate.asFile().append(content, encoding) }

        // 为Collection添加统计DSL
        Collection.metaClass.sumBy = { Closure closure -> delegate.collect(closure).sum() }
        Collection.metaGroupBy.groupAndCount = { closure -> delegate.groupBy(closure).collectEntries { k, v -> [k, v.size()] } }
        Collection.metaClass.averageBy = { closure -> delegate.collect(closure).sum() / delegate.size() }
        Collection.metaClass.maxBy = { closure -> delegate.max { a, b -> closure(a) <=> closure(b) } }
        Collection.metaClass.minBy = { closure -> delegate.min { a, b -> closure(a) <=> closure(b) } }

        // 为Number添加时间DSL
        Number.metaClass.getMilliseconds = { -> delegate * 1000L }
        Number.metaClass.getSeconds = { -> delegate * 1000L }
        Number.metaClass.getMinutes = { -> delegate * 60 * 1000L }
        Number.metaClass.getHours = { -> -> delegate * 60 * 60 * 1000L }
        Number.metaClass.getDays = { -> -> delegate * 24 * 60 * 60 * 1000L }

        // 为Date添加格式化DSL
        Date.metaClass.format = { pattern = "yyyy-MM-dd HH:mm:ss" -> delegate.format(pattern) }
        Date.metaClass.toIsoString = { -> delegate.format("yyyy-MM-dd'T'HH:mm:ss'Z'") }

        // 为Map添加深度访问DSL
        Map.metaClass.dig = { String... keys ->
            def current = delegate
            for (key in keys) {
                if (current instanceof Map && current.containsKey(key)) {
                    current = current[key]
                } else {
                    return null
                }
            }
            return current
        }

        Map.metaClass.dig = { List keys -> delegate.dig(*keys) }
    }
}

// 应用DSL增强
DSLEnhancements.enhance()

// 使用扩展的DSL
println "=== 文件操作DSL ==="
"config.txt".writeAsText("Hello, DSL World!")
println "config.txt".readAsText()
"config.txt".appendAsText("\nThis is appended content.")
println "config.txt".readAsText()

println "\n=== 集合统计DSL ==="
def numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def users = [
    [name: "Alice", age: 25, department: "Engineering"],
    [name: "Bob", age: 30, department: "Engineering"],
    [name: "Charlie", age: 35, department: "Marketing"],
    [name: "Diana", age: 28, department: "Engineering"]
]

println "Sum of numbers: ${numbers.sumBy { it }}"
println "Average age: ${users.averageBy { it.age }}"
println "Oldest person: ${users.maxBy { it.age }.name}"
println "Department counts: ${users.groupAndCount { it.department }}"

println "\n=== 时间DSL ==="
println "5 seconds in milliseconds: ${5.seconds.milliseconds}"
println "2 minutes in milliseconds: ${2.minutes.milliseconds}"
println "1 day in milliseconds: ${1.day.milliseconds}"

println "\n=== 日期格式化DSL ==="
def now = new Date()
println "Formatted date: ${now.format()}"
println "ISO date: ${now.toIsoString()}"

println "\n=== 深度访问DSL ==="
def nestedData = [
    server: [
        database: [
            primary: [
                host: "db1.example.com",
                port: 5432
            ],
            replica: [
                host: "db2.example.com",
                port: 5432
            ]
        ]
    ]
]

println "Primary DB host: ${nestedData.dig('server', 'database', 'primary', 'host')}"
println "Replica DB port: ${nestedData.dig(['server', 'database', 'replica', 'port'])}"
```

### 11.4.2 Category模式

Category模式提供了一种更结构化的方式来添加DSL方法。

```groovy
// Category模式：时间处理DSL
class TimeDSL {
    static Date getDayFromNow(Integer days) {
        new Date() + days
    }

    static Date getHourFromNow(Integer hours) {
        new Date() + (hours / 24)
    }

    static Date getMinuteFromNow(Integer minutes) {
        new Date() + (minutes / (24 * 60))
    }

    static Date getSecondsFromNow(Integer seconds) {
        new Date() + (seconds / (24 * 60 * 60))
    }

    static String getFormattedDate(Date date) {
        date.format("yyyy-MM-dd HH:mm:ss")
    }

    static Date getBeginningOfDay(Date date) {
        Calendar cal = Calendar.instance
        cal.time = date
        cal.set(Calendar.HOUR_OF_DAY, 0)
        cal.set(Calendar.MINUTE, 0)
        cal.set(Calendar.SECOND, 0)
        cal.set(Calendar.MILLISECOND, 0)
        cal.time
    }

    static Date getEndOfDay(Date date) {
        Calendar cal = Calendar.instance
        cal.time = date
        cal.set(Calendar.HOUR_OF_DAY, 23)
        cal.set(Calendar.MINUTE, 59)
        cal.set(Calendar.SECOND, 59)
        cal.set(Calendar.MILLISECOND, 999)
        cal.time
    }
}

// 集合处理DSL
class CollectionDSL {
    static List takeWhile(Collection collection, Closure condition) {
        def result = []
        for (item in collection) {
            if (condition(item)) {
                result.add(item)
            } else {
                break
            }
        }
        result
    }

    static List takeUntil(Collection collection, Closure condition) {
        def result = []
        for (item in collection) {
            if (condition(item)) break
            result.add(item)
        }
        result
    }

    static Map groupByKey(Collection collection, String key) {
        collection.groupBy { it."${key}" }
    }

    static List sortByField(Collection collection, String field, String direction = "ASC") {
        def isAscending = direction.toUpperCase() == "ASC"
        collection.sort { a, b ->
            def compare = a."${field}" <=> b."${field}"
            isAscending ? compare : -compare
        }
    }

    static Map sumByField(Collection collection, String field) {
        collection.groupBy { it."${field}" }
                .collectEntries { key, values -> [key, values.sum { it."${field}" }] }
    }
}

// 使用Category模式
use(TimeDSL) {
    println "=== 时间DSL ==="
    def tomorrow = 1.dayFromNow
    def nextWeek = 7.daysFromNow
    def inTwoHours = 2.hoursFromNow

    println "Tomorrow: ${tomorrow.formattedDate}"
    println "Next week: ${nextWeek.formattedDate}"
    println "In 2 hours: ${inTwoHours.formattedDate}"

    def today = new Date()
    println "Beginning of today: ${today.beginningOfDay.formattedDate}"
    println "End of today: ${today.endOfDay.formattedDate}"
}

use(CollectionDSL) {
    println "\n=== 集合DSL ==="
    def numbers = [1, 2, 3, 4, 5, 1, 2, 3]

    println "Take while < 4: ${numbers.takeWhile { it < 4 }}"
    println "Take until > 3: ${numbers.takeUntil { it > 3 }}"

    def people = [
        [name: "Alice", age: 25, department: "Engineering", salary: 80000],
        [name: "Bob", age: 30, department: "Marketing", salary: 75000],
        [name: "Charlie", age: 35, department: "Engineering", salary: 90000],
        [name: "Diana", age: 28, department: "Marketing", salary: 70000]
    ]

    println "Grouped by department: ${people.groupByKey('department')}"
    println "Sorted by age: ${people.sortByField('age')}"
    println "Sum by department (salary): ${people.sumByField('department')}"
}
```

## 11.5 配置DSL模式

### 11.5.1 属性映射模式

属性映射模式允许将外部配置映射到对象属性。

```groovy
// 属性映射模式：配置属性DSL
class PropertyMapper {
    private Map<String, Object> properties = [:]

    def propertyMissing(String name, Object value) {
        properties[name] = value
    }

    def propertyMissing(String name) {
        properties[name]
    }

    def map(Object target) {
        properties.each { key, value ->
            if (target.metaClass.hasProperty(target, key)) {
                target."${key}" = value
            }
        }
        return target
    }

    def mapWithTransform(Object target, Closure<String> transform) {
        properties.each { key, value ->
            def transformedKey = transform(key)
            if (target.metaClass.hasProperty(target, transformedKey)) {
                target."${transformedKey}" = value
            }
        }
        return target
    }

    def toMap() {
        return properties.clone()
    }

    def toProperties() {
        def props = new Properties()
        properties.each { key, value ->
            props.setProperty(key, value.toString())
        }
        return props
    }

    void saveToFile(String filename) {
        def props = toProperties()
        new File(filename).withOutputStream { stream ->
            props.store(stream, "Generated by PropertyMapper")
        }
    }

    static PropertyMapper loadFromFile(String filename) {
        def props = new Properties()
        new File(filename).withInputStream { stream ->
            props.load(stream)
        }

        def mapper = new PropertyMapper()
        props.each { key, value ->
            mapper."${key}" = value
        }
        return mapper
    }
}

// 配置类
class DatabaseConfig {
    String url
    String username
    String password
    int maxConnections = 10
    int timeout = 30000
    boolean autoReconnect = true

    String toString() {
        """DatabaseConfig[
            url=${url},
            username=${username},
            password=${password?.replaceAll('.', '*')},
            maxConnections=${maxConnections},
            timeout=${timeout},
            autoReconnect=${autoReconnect}
        ]"""
    }
}

class ServerConfig {
    String host = "localhost"
    int port = 8080
    String protocol = "HTTP"
    boolean sslEnabled = false
    int threadPoolSize = 50

    String toString() {
        """ServerConfig[
            host=${host},
            port=${port},
            protocol=${protocol},
            sslEnabled=${sslEnabled},
            threadPoolSize=${threadPoolSize}
        ]"""
    }
}

// 使用属性映射模式
def mapper = new PropertyMapper()

// 配置数据库
mapper {
    url "jdbc:mysql://localhost:3306/myapp"
    username "appuser"
    password "apppassword"
    max_connections 20
    connection_timeout 45000
    auto_reconnect true
}

// 配置服务器
mapper {
    server_host "api.example.com"
    server_port 443
    server_protocol "HTTPS"
    server_ssl true
    server_threads 100
}

// 转换和映射到对象
def dbConfig = mapper.mapWithTransform(new DatabaseConfig()) { key ->
    key.replace('_', '').toLowerCase()
}

def serverConfig = mapper.mapWithTransform(new ServerConfig()) { key ->
    if (key.startsWith('server_')) {
        key.substring(7)
    } else {
        key.replace('_', '').toLowerCase()
    }
}

println "Database Configuration:"
println dbConfig

println "\nServer Configuration:"
println serverConfig

// 保存和加载配置
mapper.saveToFile("config.properties")
println "\nConfiguration saved to config.properties"

def loadedMapper = PropertyMapper.loadFromFile("config.properties")
println "\nLoaded configuration: ${loadedMapper.toMap()}"
```

### 11.5.2 环境特定配置

```groovy
// 环境特定配置DSL
class EnvironmentConfig {
    private Map<String, Map<String, Object>> environments = [:]
    private String currentEnvironment = "default"

    def environment(String name, Closure closure) {
        def builder = new ConfigBuilder()
        closure.delegate = builder
        closure.resolveStrategy = Closure.DELEGATE_FIRST
        closure()
        environments[name] = builder.config
        return this
    }

    def setCurrent(String env) {
        if (environments.containsKey(env)) {
            currentEnvironment = env
        } else {
            throw new IllegalArgumentException("Unknown environment: ${env}")
        }
        return this
    }

    def getCurrent() {
        return environments[currentEnvironment] ?: [:]
    }

    def get(String key, Object defaultValue = null) {
        def current = getCurrent()
        def value = current[key]

        if (value != null) {
            return value
        }

        // 如果在当前环境找不到，尝试从default环境查找
        if (currentEnvironment != "default") {
            def defaultEnv = environments["default"]
            if (defaultEnv) {
                return defaultEnv[key] ?: defaultValue
            }
        }

        return defaultValue
    }

    def getAll() {
        def result = environments["default"] ?: [:]
        def current = getCurrent()
        result.putAll(current)
        return result
    }

    def merge(String... envNames) {
        def merged = [:]
        envNames.each { envName ->
            if (environments.containsKey(envName)) {
                merged.putAll(environments[envName])
            }
        }
        return merged
    }
}

class ConfigBuilder {
    Map<String, Object> config = [:]

    def methodMissing(String name, args) {
        if (args && args[0] != null) {
            config[name] = args[0]
            return this
        }
        return config[name]
    }

    def propertyMissing(String name, Object value) {
        config[name] = value
    }

    def propertyMissing(String name) {
        config[name]
    }

    def nested(String name, Closure closure) {
        def builder = new ConfigBuilder()
        closure.delegate = builder
        closure.resolveStrategy = Closure.DELEGATE_FIRST
        closure()
        config[name] = builder.config
        return this
    }
}

// 使用环境特定配置DSL
def envConfig = new EnvironmentConfig()

envConfig.environment("default") {
    app_name "My Application"
    app_version "1.0.0"
    log_level "INFO"

    database {
        driver "com.mysql.jdbc.Driver"
        port 3306
        timeout 30000
    }

    server {
        host "localhost"
        port 8080
        threads 50
    }
}

envConfig.environment("development") {
    log_level "DEBUG"

    database {
        host "localhost"
        database "myapp_dev"
        username "devuser"
        password "devpass"
        max_connections 10
    }

    server {
        ssl_enabled false
        debug true
    }

    features {
        enable_profiler true
        enable_mock_services true
    }
}

envConfig.environment("production") {
    log_level "WARN"

    database {
        host "prod-db.example.com"
        database "myapp_prod"
        username "produser"
        password System.getenv("DB_PASSWORD")
        max_connections 50
        ssl_enabled true
    }

    server {
        host "app.example.com"
        port 443
        ssl_enabled true
        threads 200
    }

    features {
        enable_profiler false
        enable_monitoring true
        enable_caching true
    }
}

// 使用配置
println "=== Development Environment ==="
envConfig.setCurrent("development")
println "App Name: ${envConfig.get('app_name')}"
println "Database Host: ${envConfig.get('database.host')}"
println "SSL Enabled: ${envConfig.get('server.ssl_enabled')}"
println "Profiler Enabled: ${envConfig.get('features.enable_profiler')}"

println "\n=== Production Environment ==="
envConfig.setCurrent("production")
println "App Name: ${envConfig.get('app_name')}"
println "Database Host: ${envConfig.get('database.host')}"
println "SSL Enabled: ${envConfig.get('server.ssl_enabled')}"
println "Profiler Enabled: ${envConfig.get('features.enable_profiler')}"

println "\n=== Current Configuration ==="
println envConfig.getAll()

println "\n=== Merged Configuration ==="
def merged = envConfig.merge("default", "production")
println merged
```

## 本章小结

Groovy DSL设计模式提供了丰富的工具和方法来创建优雅、易用的领域特定语言。

### 核心设计模式回顾

1. **Builder模式**：链式调用构建复杂对象
2. **Command Chain模式**：流畅的方法链接口
3. **嵌套闭包模式**：层次化配置结构
4. **扩展方法模式**：为现有类添加DSL功能
5. **配置DSL模式**：灵活的配置映射机制

### 实战应用

✅ **掌握Builder模式**：创建链式调用的DSL接口
✅ **学会Command Chain模式**：实现流畅的方法调用链
✅ **理解嵌套闭包**：构建层次化的配置结构
✅ **运用扩展方法**：增强现有类的DSL能力
✅ **设计配置DSL**：实现灵活的配置管理系统

### 设计原则

- **简洁性优先**：DSL应该易于阅读和理解
- **类型安全**：尽可能提供编译时类型检查
- **错误友好**：提供清晰的错误信息和反馈
- **性能考虑**：避免DSL引入过大的性能开销
- **扩展性**：设计时考虑未来的扩展需求

### 最佳实践

- **保持一致性**：命名、结构、行为的一致性
- **避免过度设计**：不要为了DSL而DSL
- **提供文档**：为DSL提供清晰的使用说明
- **测试覆盖**：确保DSL的正确性和稳定性
- **工具支持**：考虑IDE支持和代码补全

下一章我们将深入探讨闭包委托策略，这是Groovy DSL开发的核心机制之一。