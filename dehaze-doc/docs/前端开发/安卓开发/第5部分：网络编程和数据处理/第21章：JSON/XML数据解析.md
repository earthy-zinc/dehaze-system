# 第21章：JSON/XML数据解析

## 📦 Gson库使用详解

### Gson基础使用

Gson是Google提供的用于JSON数据序列化和反序列化的Java库，它提供了简单易用的API来处理JSON数据。

#### 基本序列化和反序列化

```java
// GsonBasicExample.java
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class GsonBasicExample {

    // 用户数据模型
    public static class User {
        private String name;
        private int age;
        private String email;
        private boolean active;

        public User() {}

        public User(String name, int age, String email, boolean active) {
            this.name = name;
            this.age = age;
            this.email = email;
            this.active = active;
        }

        // Getters and Setters
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public int getAge() { return age; }
        public void setAge(int age) { this.age = age; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public boolean isActive() { return active; }
        public void setActive(boolean active) { this.active = active; }

        @Override
        public String toString() {
            return String.format("User{name='%s', age=%d, email='%s', active=%s}",
                name, age, email, active);
        }
    }

    public static void main(String[] args) {
        // 创建Gson实例
        Gson gson = new Gson();

        // 1. 序列化：Java对象 -> JSON字符串
        User user = new User("张三", 25, "zhangsan@example.com", true);
        String json = gson.toJson(user);
        System.out.println("序列化结果: " + json);

        // 2. 反序列化：JSON字符串 -> Java对象
        String jsonString = "{\"name\":\"李四\",\"age\":30,\"email\":\"lisi@example.com\",\"active\":false}";
        User deserializedUser = gson.fromJson(jsonString, User.class);
        System.out.println("反序列化结果: " + deserializedUser);

        // 3. 处理数组
        User[] users = {
            new User("王五", 28, "wangwu@example.com", true),
            new User("赵六", 32, "zhaoliu@example.com", false)
        };
        String usersJson = gson.toJson(users);
        System.out.println("数组序列化: " + usersJson);

        // 4. 处理List集合
        List<User> userList = Arrays.asList(users);
        Type userListType = new TypeToken<List<User>>(){}.getType();
        String listJson = gson.toJson(userList);
        System.out.println("List序列化: " + listJson);

        // 5. 反序列化List
        List<User> deserializedList = gson.fromJson(listJson, userListType);
        System.out.println("List反序列化: " + deserializedList);

        // 6. 处理Map
        Map<String, Object> dataMap = Map.of(
            "name", "测试用户",
            "age", 25,
            "active", true,
            "scores", new int[]{85, 90, 78}
        );
        String mapJson = gson.toJson(dataMap);
        System.out.println("Map序列化: " + mapJson);

        // 7. 反序列化Map
        Type mapType = new TypeToken<Map<String, Object>>(){}.getType();
        Map<String, Object> deserializedMap = gson.fromJson(mapJson, mapType);
        System.out.println("Map反序列化: " + deserializedMap);
    }
}
```

### Gson高级配置

```java
// GsonAdvancedExample.java
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.annotations.SerializedName;
import com.google.gson.reflect.TypeToken;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;

import java.lang.reflect.Type;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Map;

public class GsonAdvancedExample {

    // 复杂数据模型
    public static class Product {
        @SerializedName("product_id")
        private int id;

        @SerializedName("product_name")
        private String name;

        @SerializedName("product_price")
        private double price;

        @SerializedName("in_stock")
        private boolean inStock;

        @SerializedName(value = "product_description", alternate = {"desc", "product_desc"})
        private String description;

        @SerializedName("created_date")
        private Date createdDate;

        private transient String internalId; // transient字段不会被序列化

        public Product() {}

        public Product(int id, String name, double price, boolean inStock, String description, Date createdDate) {
            this.id = id;
            this.name = name;
            this.price = price;
            this.inStock = inStock;
            this.description = description;
            this.createdDate = createdDate;
            this.internalId = "PROD_" + System.currentTimeMillis();
        }

        // Getters and Setters
        public int getId() { return id; }
        public void setId(int id) { this.id = id; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public double getPrice() { return price; }
        public void setPrice(double price) { this.price = price; }
        public boolean isInStock() { return inStock; }
        public void setInStock(boolean inStock) { this.inStock = inStock; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public Date getCreatedDate() { return createdDate; }
        public void setCreatedDate(Date createdDate) { this.createdDate = createdDate; }
        public String getInternalId() { return internalId; }

        @Override
        public String toString() {
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault());
            return String.format("Product{id=%d, name='%s', price=%.2f, inStock=%s, description='%s', createdDate=%s, internalId='%s'}",
                id, name, price, inStock, description,
                createdDate != null ? sdf.format(createdDate) : "null", internalId);
        }
    }

    // 自定义日期序列化器
    public static class DateSerializer implements JsonSerializer<Date> {
        private final SimpleDateFormat dateFormat;

        public DateSerializer(String pattern) {
            this.dateFormat = new SimpleDateFormat(pattern, Locale.getDefault());
        }

        @Override
        public JsonElement serialize(Date src, Type typeOfSrc, JsonSerializationContext context) {
            return new JsonPrimitive(dateFormat.format(src));
        }
    }

    // 自定义日期反序列化器
    public static class DateDeserializer implements JsonDeserializer<Date> {
        private final SimpleDateFormat dateFormat;

        public DateDeserializer(String pattern) {
            this.dateFormat = new SimpleDateFormat(pattern, Locale.getDefault());
        }

        @Override
        public Date deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context)
                throws JsonParseException {
            try {
                return dateFormat.parse(json.getAsString());
            } catch (ParseException e) {
                throw new JsonParseException("日期解析失败: " + json.getAsString(), e);
            }
        }
    }

    public static void main(String[] args) {
        // 1. 创建高级Gson配置
        Gson gson = new GsonBuilder()
                .setPrettyPrinting() // 格式化输出
                .serializeNulls() // 序列化null值
                .disableHtmlEscaping() // 禁用HTML转义
                .setDateFormat("yyyy-MM-dd HH:mm:ss") // 设置日期格式
                .create();

        // 2. 使用自定义序列化器
        Gson customGson = new GsonBuilder()
                .registerTypeAdapter(Date.class, new DateSerializer("yyyy-MM-dd"))
                .registerTypeAdapter(Date.class, new DateDeserializer("yyyy-MM-dd"))
                .setFieldNamingStrategy(field -> {
                    // 自定义字段命名策略
                    switch (field.getName()) {
                        case "id": return "product_id";
                        case "name": return "product_name";
                        case "price": return "product_price";
                        case "inStock": return "in_stock";
                        case "description": return "product_description";
                        case "createdDate": return "created_date";
                        default: return field.getName();
                    }
                })
                .setPrettyPrinting()
                .create();

        // 3. 测试序列化
        Date now = new Date();
        Product product = new Product(1001, "智能手机", 2999.99, true, "高性能智能手机", now);

        String json = gson.toJson(product);
        System.out.println("标准Gson序列化结果:");
        System.out.println(json);

        String customJson = customGson.toJson(product);
        System.out.println("\n自定义Gson序列化结果:");
        System.out.println(customJson);

        // 4. 测试反序列化
        String testJson = "{\"product_id\":1002,\"product_name\":\"笔记本电脑\",\"product_price\":5999.99,\"in_stock\":true,\"product_description\":\"商务办公笔记本\",\"created_date\":\"2024-01-15\"}";

        Product deserializedProduct = customGson.fromJson(testJson, Product.class);
        System.out.println("\n反序列化结果:");
        System.out.println(deserializedProduct);

        // 5. 处理复杂嵌套对象
        testComplexObjectSerialization();
    }

    private static void testComplexObjectSerialization() {
        // 复杂嵌套对象测试
        Map<String, Object> complexData = Map.of(
            "user", Map.of("name", "张三", "age", 25),
            "orders", new Object[]{
                Map.of("id", 1001, "amount", 299.99, "items", new String[]{"商品A", "商品B"}),
                Map.of("id", 1002, "amount", 599.99, "items", new String[]{"商品C"})
            },
            "metadata", Map.of("version", "1.0", "timestamp", System.currentTimeMillis())
        );

        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String complexJson = gson.toJson(complexData);
        System.out.println("\n复杂对象序列化结果:");
        System.out.println(complexJson);
    }
}
```

## 🚀 Moshi高级特性

### Moshi基础使用

Moshi是Square公司开发的另一个优秀的JSON解析库，相比Gson更加现代化和高效。

```java
// MoshiBasicExample.java
import com.squareup.moshi.Json;
import com.squareup.moshi.JsonAdapter;
import com.squareup.moshi.Moshi;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public class MoshiBasicExample {

    // 用户数据模型（使用Moshi注解）
    public static class User {
        @Json(name = "user_id")
        private int id;

        @Json(name = "user_name")
        private String name;

        @Json(name = "user_email")
        private String email;

        @Json(name = "is_active")
        private boolean active;

        public User() {}

        public User(int id, String name, String email, boolean active) {
            this.id = id;
            this.name = name;
            this.email = email;
            this.active = active;
        }

        // Getters and Setters
        public int getId() { return id; }
        public void setId(int id) { this.id = id; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public boolean isActive() { return active; }
        public void setActive(boolean active) { this.active = active; }

        @Override
        public String toString() {
            return String.format("User{id=%d, name='%s', email='%s', active=%s}",
                id, name, email, active);
        }
    }

    public static void main(String[] args) {
        // 创建Moshi实例
        Moshi moshi = new Moshi.Builder().build();

        // 创建JsonAdapter
        JsonAdapter<User> userAdapter = moshi.adapter(User.class);

        try {
            // 1. 序列化：Java对象 -> JSON字符串
            User user = new User(1, "张三", "zhangsan@example.com", true);
            String json = userAdapter.toJson(user);
            System.out.println("Moshi序列化结果: " + json);

            // 2. 反序列化：JSON字符串 -> Java对象
            String jsonString = "{\"user_id\":2,\"user_name\":\"李四\",\"user_email\":\"lisi@example.com\",\"is_active\":false}";
            User deserializedUser = userAdapter.fromJson(jsonString);
            System.out.println("Moshi反序列化结果: " + deserializedUser);

            // 3. 处理List
            JsonAdapter<List<User>> userListAdapter = moshi.adapter(
                Types.newParameterizedType(List.class, User.class));

            String listJson = "[{\"user_id\":3,\"user_name\":\"王五\",\"user_email\":\"wangwu@example.com\",\"is_active\":true}]";
            List<User> userList = userListAdapter.fromJson(listJson);
            System.out.println("List反序列化结果: " + userList);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

### Moshi自定义适配器

```java
// MoshiCustomAdapter.java
import com.squareup.moshi.FromJson;
import com.squareup.moshi.JsonAdapter;
import com.squareup.moshi.JsonReader;
import com.squareup.moshi.JsonWriter;
import com.squareup.moshi.Moshi;
import com.squareup.moshi.ToJson;
import com.squareup.moshi.Types;

import java.io.IOException;
import java.lang.reflect.Type;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MoshiCustomAdapter {

    // 日期适配器
    public static class DateAdapter {
        private final SimpleDateFormat dateFormat;

        public DateAdapter(String pattern) {
            this.dateFormat = new SimpleDateFormat(pattern, Locale.getDefault());
        }

        @ToJson
        String toJson(Date date) {
            return dateFormat.format(date);
        }

        @FromJson
        Date fromJson(String json) throws IOException {
            try {
                return dateFormat.parse(json);
            } catch (ParseException e) {
                throw new IOException("日期解析失败: " + json, e);
            }
        }
    }

    // 枚举适配器
    public enum Status {
        ACTIVE("active"),
        INACTIVE("inactive"),
        PENDING("pending");

        private final String value;

        Status(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        public static Status fromValue(String value) {
            for (Status status : Status.values()) {
                if (status.value.equals(value)) {
                    return status;
                }
            }
            throw new IllegalArgumentException("未知的Status值: " + value);
        }
    }

    public static class StatusAdapter {
        @ToJson
        String toJson(Status status) {
            return status.getValue();
        }

        @FromJson
        Status fromJson(String json) {
            return Status.fromValue(json);
        }
    }

    // 复杂数据模型
    public static class Order {
        private int id;
        private String orderNumber;
        private double amount;
        private Date orderDate;
        private Status status;

        public Order() {}

        public Order(int id, String orderNumber, double amount, Date orderDate, Status status) {
            this.id = id;
            this.orderNumber = orderNumber;
            this.amount = amount;
            this.orderDate = orderDate;
            this.status = status;
        }

        // Getters and Setters
        public int getId() { return id; }
        public void setId(int id) { this.id = id; }
        public String getOrderNumber() { return orderNumber; }
        public void setOrderNumber(String orderNumber) { this.orderNumber = orderNumber; }
        public double getAmount() { return amount; }
        public void setAmount(double amount) { this.amount = amount; }
        public Date getOrderDate() { return orderDate; }
        public void setOrderDate(Date orderDate) { this.orderDate = orderDate; }
        public Status getStatus() { return status; }
        public void setStatus(Status status) { this.status = status; }

        @Override
        public String toString() {
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault());
            return String.format("Order{id=%d, orderNumber='%s', amount=%.2f, orderDate=%s, status=%s}",
                id, orderNumber, amount, orderDate != null ? sdf.format(orderDate) : "null", status);
        }
    }

    public static void main(String[] args) {
        // 创建Moshi实例并注册自定义适配器
        Moshi moshi = new Moshi.Builder()
                .add(new DateAdapter("yyyy-MM-dd HH:mm:ss"))
                .add(new StatusAdapter())
                .build();

        // 创建JsonAdapter
        JsonAdapter<Order> orderAdapter = moshi.adapter(Order.class);

        try {
            // 1. 序列化
            Date now = new Date();
            Order order = new Order(1001, "ORD-2024-001", 2999.99, now, Status.ACTIVE);
            String json = orderAdapter.toJson(order);
            System.out.println("自定义适配器序列化结果:");
            System.out.println(json);

            // 2. 反序列化
            String jsonString = "{\"id\":1002,\"orderNumber\":\"ORD-2024-002\",\"amount\":5999.99,\"orderDate\":\"2024-01-15 14:30:00\",\"status\":\"pending\"}";
            Order deserializedOrder = orderAdapter.fromJson(jsonString);
            System.out.println("\n自定义适配器反序列化结果:");
            System.out.println(deserializedOrder);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

## 📄 XML解析技术

### DOM解析器

DOM（Document Object Model）解析器将整个XML文档加载到内存中，形成树形结构，适合处理较小的XML文件。

```java
// DOMParser.java
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class DOMParser {

    // 用户数据模型
    public static class User {
        private int id;
        private String name;
        private String email;
        private String phone;
        private List<String> hobbies;

        public User() {
            this.hobbies = new ArrayList<>();
        }

        public User(int id, String name, String email, String phone, List<String> hobbies) {
            this.id = id;
            this.name = name;
            this.email = email;
            this.phone = phone;
            this.hobbies = hobbies != null ? hobbies : new ArrayList<>();
        }

        // Getters and Setters
        public int getId() { return id; }
        public void setId(int id) { this.id = id; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getPhone() { return phone; }
        public void setPhone(String phone) { this.phone = phone; }
        public List<String> getHobbies() { return hobbies; }
        public void setHobbies(List<String> hobbies) { this.hobbies = hobbies; }

        @Override
        public String toString() {
            return String.format("User{id=%d, name='%s', email='%s', phone='%s', hobbies=%s}",
                id, name, email, phone, hobbies);
        }
    }

    public static List<User> parseUsers(String xmlString) {
        List<User> users = new ArrayList<>();

        try {
            // 创建DocumentBuilderFactory
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();

            // 安全配置，防止XML外部实体攻击
            factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
            factory.setFeature("http://xml.org/sax/features/external-general-entities", false);
            factory.setFeature("http://xml.org/sax/features/external-parameter-entities", false);
            factory.setXIncludeAware(false);
            factory.setExpandEntityReferences(false);

            // 创建DocumentBuilder
            DocumentBuilder builder = factory.newDocumentBuilder();

            // 解析XML文档
            InputStream inputStream = new ByteArrayInputStream(xmlString.getBytes("UTF-8"));
            Document document = builder.parse(inputStream);

            // 获取根元素
            Element root = document.getDocumentElement();
            root.normalize();

            // 解析用户列表
            NodeList userNodes = root.getElementsByTagName("user");

            for (int i = 0; i < userNodes.getLength(); i++) {
                Node userNode = userNodes.item(i);
                if (userNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element userElement = (Element) userNode;
                    User user = parseUserElement(userElement);
                    if (user != null) {
                        users.add(user);
                    }
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        return users;
    }

    private static User parseUserElement(Element userElement) {
        try {
            User user = new User();

            // 解析基本属性
            user.setId(Integer.parseInt(userElement.getAttribute("id")));

            // 解析子元素
            NodeList childNodes = userElement.getChildNodes();
            for (int i = 0; i < childNodes.getLength(); i++) {
                Node childNode = childNodes.item(i);
                if (childNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element childElement = (Element) childNode;
                    String tagName = childElement.getTagName();
                    String textContent = childElement.getTextContent();

                    switch (tagName) {
                        case "name":
                            user.setName(textContent);
                            break;
                        case "email":
                            user.setEmail(textContent);
                            break;
                        case "phone":
                            user.setPhone(textContent);
                            break;
                        case "hobbies":
                            parseHobbies(childElement, user);
                            break;
                    }
                }
            }

            return user;

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private static void parseHobbies(Element hobbiesElement, User user) {
        NodeList hobbyNodes = hobbiesElement.getElementsByTagName("hobby");
        for (int i = 0; i < hobbyNodes.getLength(); i++) {
            Node hobbyNode = hobbyNodes.item(i);
            if (hobbyNode.getNodeType() == Node.ELEMENT_NODE) {
                Element hobbyElement = (Element) hobbyNode;
                user.getHobbies().add(hobbyElement.getTextContent());
            }
        }
    }

    public static void main(String[] args) {
        String xmlData = """
            <?xml version="1.0" encoding="UTF-8"?>
            <users>
                <user id="1">
                    <name>张三</name>
                    <email>zhangsan@example.com</email>
                    <phone>13800138001</phone>
                    <hobbies>
                        <hobby>读书</hobby>
                        <hobby>旅游</hobby>
                        <hobby>编程</hobby>
                    </hobbies>
                </user>
                <user id="2">
                    <name>李四</name>
                    <email>lisi@example.com</email>
                    <phone>13800138002</phone>
                    <hobbies>
                        <hobby>音乐</hobby>
                        <hobby>运动</hobby>
                    </hobbies>
                </user>
            </users>
            """;

        List<User> users = parseUsers(xmlData);
        System.out.println("DOM解析结果:");
        for (User user : users) {
            System.out.println(user);
        }
    }
}
```

### SAX解析器

SAX（Simple API for XML）解析器基于事件驱动，不需要将整个文档加载到内存，适合处理大型XML文件。

```java
// SAXParser.java
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class SAXParser {

    // 用户数据模型（与DOM解析器中的User类相同）
    public static class User {
        private int id;
        private String name;
        private String email;
        private String phone;
        private List<String> hobbies;

        public User() {
            this.hobbies = new ArrayList<>();
        }

        public User(int id, String name, String email, String phone, List<String> hobbies) {
            this.id = id;
            this.name = name;
            this.email = email;
            this.phone = phone;
            this.hobbies = hobbies != null ? hobbies : new ArrayList<>();
        }

        // Getters and Setters
        public int getId() { return id; }
        public void setId(int id) { this.id = id; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getPhone() { return phone; }
        public void setPhone(String phone) { this.phone = phone; }
        public List<String> getHobbies() { return hobbies; }
        public void setHobbies(List<String> hobbies) { this.hobbies = hobbies; }

        @Override
        public String toString() {
            return String.format("User{id=%d, name='%s', email='%s', phone='%s', hobbies=%s}",
                id, name, email, phone, hobbies);
        }
    }

    // 自定义SAX处理器
    private static class UserHandler extends DefaultHandler {
        private List<User> users;
        private User currentUser;
        private StringBuilder currentValue;
        private Stack<String> elementStack;

        public UserHandler() {
            this.users = new ArrayList<>();
            this.currentValue = new StringBuilder();
            this.elementStack = new Stack<>();
        }

        public List<User> getUsers() {
            return users;
        }

        @Override
        public void startElement(String uri, String localName, String qName, Attributes attributes) {
            elementStack.push(qName);
            currentValue.setLength(0); // 重置当前值

            if ("user".equals(qName)) {
                currentUser = new User();
                String idAttr = attributes.getValue("id");
                if (idAttr != null) {
                    currentUser.setId(Integer.parseInt(idAttr));
                }
            }
        }

        @Override
        public void characters(char[] ch, int start, int length) {
            currentValue.append(ch, start, length);
        }

        @Override
        public void endElement(String uri, String localName, String qName) {
            String currentValueStr = currentValue.toString().trim();
            String parentElement = elementStack.size() > 1 ?
                elementStack.get(elementStack.size() - 2) : "";

            if (currentUser != null) {
                switch (qName) {
                    case "name":
                        currentUser.setName(currentValueStr);
                        break;
                    case "email":
                        currentUser.setEmail(currentValueStr);
                        break;
                    case "phone":
                        currentUser.setPhone(currentValueStr);
                        break;
                    case "hobby":
                        if ("hobbies".equals(parentElement)) {
                            currentUser.getHobbies().add(currentValueStr);
                        }
                        break;
                    case "user":
                        users.add(currentUser);
                        currentUser = null;
                        break;
                }
            }

            elementStack.pop();
        }
    }

    public static List<User> parseUsers(String xmlString) {
        try {
            // 创建SAXParserFactory
            SAXParserFactory factory = SAXParserFactory.newInstance();

            // 安全配置
            factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
            factory.setFeature("http://xml.org/sax/features/external-general-entities", false);
            factory.setFeature("http://xml.org/sax/features/external-parameter-entities", false);
            factory.setXIncludeAware(false);

            // 创建SAXParser
            SAXParser saxParser = factory.newSAXParser();

            // 创建自定义处理器
            UserHandler handler = new UserHandler();

            // 解析XML文档
            InputStream inputStream = new ByteArrayInputStream(xmlString.getBytes("UTF-8"));
            saxParser.parse(inputStream, handler);

            return handler.getUsers();

        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    public static void main(String[] args) {
        String xmlData = """
            <?xml version="1.0" encoding="UTF-8"?>
            <users>
                <user id="1">
                    <name>张三</name>
                    <email>zhangsan@example.com</email>
                    <phone>13800138001</phone>
                    <hobbies>
                        <hobby>读书</hobby>
                        <hobby>旅游</hobby>
                        <hobby>编程</hobby>
                    </hobbies>
                </user>
                <user id="2">
                    <name>李四</name>
                    <email>lisi@example.com</email>
                    <phone>13800138002</phone>
                    <hobbies>
                        <hobby>音乐</hobby>
                        <hobby>运动</hobby>
                    </hobbies>
                </user>
            </users>
            """;

        List<User> users = parseUsers(xmlData);
        System.out.println("SAX解析结果:");
        for (User user : users) {
            System.out.println(user);
        }
    }
}
```

## 🔄 数据绑定和转换

### 统一数据解析接口

```java
// DataParser.java
import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.squareup.moshi.JsonAdapter;
import com.squareup.moshi.Moshi;
import com.squareup.moshi.Types;

import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;

public class DataParser {
    private static final String TAG = "DataParser";

    // 解析器类型枚举
    public enum ParserType {
        GSON,
        MOSHI
    }

    // 解析配置
    public static class ParserConfig {
        private ParserType type;
        private boolean prettyPrint;
        private boolean serializeNulls;
        private String dateFormat;

        public ParserConfig(ParserType type) {
            this.type = type;
            this.prettyPrint = false;
            this.serializeNulls = false;
            this.dateFormat = "yyyy-MM-dd HH:mm:ss";
        }

        // 预定义配置
        public static ParserConfig getDefaultGsonConfig() {
            return new ParserConfig(ParserType.GSON)
                    .setPrettyPrint(true)
                    .setSerializeNulls(true);
        }

        public static ParserConfig getDefaultMoshiConfig() {
            return new ParserConfig(ParserType.MOSHI);
        }

        // Builder方法
        public ParserConfig setPrettyPrint(boolean prettyPrint) {
            this.prettyPrint = prettyPrint;
            return this;
        }

        public ParserConfig setSerializeNulls(boolean serializeNulls) {
            this.serializeNulls = serializeNulls;
            return this;
        }

        public ParserConfig setDateFormat(String dateFormat) {
            this.dateFormat = dateFormat;
            return this;
        }

        // Getters
        public ParserType getType() { return type; }
        public boolean isPrettyPrint() { return prettyPrint; }
        public boolean isSerializeNulls() { return serializeNulls; }
        public String getDateFormat() { return dateFormat; }
    }

    // 数据解析器接口
    public interface IDataParser {
        <T> String toJson(T object);
        <T> T fromJson(String json, Class<T> clazz);
        <T> T fromJson(String json, Type type);
        <T> String toJsonList(List<T> list);
        <T> List<T> fromJsonList(String json, Class<T> clazz);
    }

    // Gson解析器实现
    private static class GsonParser implements IDataParser {
        private final Gson gson;

        public GsonParser(ParserConfig config) {
            GsonBuilder builder = new GsonBuilder()
                    .setDateFormat(config.getDateFormat());

            if (config.isPrettyPrint()) {
                builder.setPrettyPrinting();
            }

            if (config.isSerializeNulls()) {
                builder.serializeNulls();
            }

            this.gson = builder.create();
        }

        @Override
        public <T> String toJson(T object) {
            try {
                return gson.toJson(object);
            } catch (Exception e) {
                Log.e(TAG, "Gson序列化失败", e);
                return null;
            }
        }

        @Override
        public <T> T fromJson(String json, Class<T> clazz) {
            try {
                return gson.fromJson(json, clazz);
            } catch (Exception e) {
                Log.e(TAG, "Gson反序列化失败", e);
                return null;
            }
        }

        @Override
        public <T> T fromJson(String json, Type type) {
            try {
                return gson.fromJson(json, type);
            } catch (Exception e) {
                Log.e(TAG, "Gson反序列化失败", e);
                return null;
            }
        }

        @Override
        public <T> String toJsonList(List<T> list) {
            try {
                return gson.toJson(list);
            } catch (Exception e) {
                Log.e(TAG, "Gson列表序列化失败", e);
                return null;
            }
        }

        @Override
        public <T> List<T> fromJsonList(String json, Class<T> clazz) {
            try {
                Type listType = com.google.gson.reflect.TypeToken.getParameterizedType(List.class, clazz);
                return gson.fromJson(json, listType);
            } catch (Exception e) {
                Log.e(TAG, "Gson列表反序列化失败", e);
                return null;
            }
        }
    }

    // Moshi解析器实现
    private static class MoshiParser implements IDataParser {
        private final Moshi moshi;

        public MoshiParser(ParserConfig config) {
            Moshi.Builder builder = new Moshi.Builder();
            // 这里可以添加自定义适配器
            this.moshi = builder.build();
        }

        @Override
        public <T> String toJson(T object) {
            try {
                JsonAdapter<T> adapter = moshi.adapter((Class<T>) object.getClass());
                return adapter.toJson(object);
            } catch (Exception e) {
                Log.e(TAG, "Moshi序列化失败", e);
                return null;
            }
        }

        @Override
        public <T> T fromJson(String json, Class<T> clazz) {
            try {
                JsonAdapter<T> adapter = moshi.adapter(clazz);
                return adapter.fromJson(json);
            } catch (Exception e) {
                Log.e(TAG, "Moshi反序列化失败", e);
                return null;
            }
        }

        @Override
        public <T> T fromJson(String json, Type type) {
            try {
                JsonAdapter<T> adapter = moshi.adapter(type);
                return adapter.fromJson(json);
            } catch (Exception e) {
                Log.e(TAG, "Moshi反序列化失败", e);
                return null;
            }
        }

        @Override
        public <T> String toJsonList(List<T> list) {
            try {
                Type listType = Types.newParameterizedType(List.class, Object.class);
                JsonAdapter<List<T>> adapter = moshi.adapter(listType);
                return adapter.toJson(list);
            } catch (Exception e) {
                Log.e(TAG, "Moshi列表序列化失败", e);
                return null;
            }
        }

        @Override
        public <T> List<T> fromJsonList(String json, Class<T> clazz) {
            try {
                Type listType = Types.newParameterizedType(List.class, clazz);
                JsonAdapter<List<T>> adapter = moshi.adapter(listType);
                return adapter.fromJson(json);
            } catch (Exception e) {
                Log.e(TAG, "Moshi列表反序列化失败", e);
                return null;
            }
        }
    }

    // 创建解析器
    public static IDataParser createParser(ParserConfig config) {
        switch (config.getType()) {
            case GSON:
                return new GsonParser(config);
            case MOSHI:
                return new MoshiParser(config);
            default:
                return new GsonParser(ParserConfig.getDefaultGsonConfig());
        }
    }

    // 创建默认Gson解析器
    public static IDataParser createGsonParser() {
        return createParser(ParserConfig.getDefaultGsonConfig());
    }

    // 创建默认Moshi解析器
    public static IDataParser createMoshiParser() {
        return createParser(ParserConfig.getDefaultMoshiConfig());
    }

    // 数据转换工具类
    public static class DataConverter {
        private final IDataParser parser;

        public DataConverter(IDataParser parser) {
            this.parser = parser;
        }

        // 对象转Map
        public Map<String, Object> objectToMap(Object object) {
            try {
                String json = parser.toJson(object);
                return parser.fromJson(json, Map.class);
            } catch (Exception e) {
                Log.e(TAG, "对象转Map失败", e);
                return null;
            }
        }

        // Map转对象
        public <T> T mapToObject(Map<String, Object> map, Class<T> clazz) {
            try {
                String json = parser.toJson(map);
                return parser.fromJson(json, clazz);
            } catch (Exception e) {
                Log.e(TAG, "Map转对象失败", e);
                return null;
            }
        }

        // 深度复制对象
        public <T> T deepCopy(T object, Class<T> clazz) {
            try {
                String json = parser.toJson(object);
                return parser.fromJson(json, clazz);
            } catch (Exception e) {
                Log.e(TAG, "对象深度复制失败", e);
                return null;
            }
        }

        // 比较两个对象是否相等（通过JSON字符串比较）
        public boolean equals(Object obj1, Object obj2) {
            try {
                String json1 = parser.toJson(obj1);
                String json2 = parser.toJson(obj2);
                return json1.equals(json2);
            } catch (Exception e) {
                Log.e(TAG, "对象比较失败", e);
                return false;
            }
        }

        // 合并两个对象（简单实现，相同字段后者覆盖前者）
        public <T> T mergeObjects(T obj1, T obj2, Class<T> clazz) {
            try {
                Map<String, Object> map1 = objectToMap(obj1);
                Map<String, Object> map2 = objectToMap(obj2);

                if (map1 != null && map2 != null) {
                    map1.putAll(map2);
                    return mapToObject(map1, clazz);
                }
                return null;
            } catch (Exception e) {
                Log.e(TAG, "对象合并失败", e);
                return null;
            }
        }
    }

    // 使用示例
    public static void main(String[] args) {
        // 测试数据
        TestData data = new TestData(1, "测试数据", 25);

        // 1. 使用Gson解析器
        IDataParser gsonParser = createGsonParser();
        String gsonJson = gsonParser.toJson(data);
        System.out.println("Gson序列化结果: " + gsonJson);

        TestData gsonData = gsonParser.fromJson(gsonJson, TestData.class);
        System.out.println("Gson反序列化结果: " + gsonData);

        // 2. 使用Moshi解析器
        IDataParser moshiParser = createMoshiParser();
        String moshiJson = moshiParser.toJson(data);
        System.out.println("Moshi序列化结果: " + moshiJson);

        TestData moshiData = moshiParser.fromJson(moshiJson, TestData.class);
        System.out.println("Moshi反序列化结果: " + moshiData);

        // 3. 使用数据转换器
        DataConverter converter = new DataConverter(gsonParser);
        Map<String, Object> map = converter.objectToMap(data);
        System.out.println("对象转Map: " + map);

        TestData convertedData = converter.mapToObject(map, TestData.class);
        System.out.println("Map转对象: " + convertedData);

        TestData copiedData = converter.deepCopy(data, TestData.class);
        System.out.println("深度复制: " + copiedData);

        boolean isEqual = converter.equals(data, copiedData);
        System.out.println("对象相等性比较: " + isEqual);
    }

    // 测试数据类
    public static class TestData {
        private int id;
        private String name;
        private int age;

        public TestData() {}

        public TestData(int id, String name, int age) {
            this.id = id;
            this.name = name;
            this.age = age;
        }

        // Getters and Setters
        public int getId() { return id; }
        public void setId(int id) { this.id = id; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public int getAge() { return age; }
        public void setAge(int age) { this.age = age; }

        @Override
        public String toString() {
            return String.format("TestData{id=%d, name='%s', age=%d}", id, name, age);
        }
    }
}
```

## ⚡ 性能优化技巧

### 解析性能优化

```java
// PerformanceOptimization.java
import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class PerformanceOptimization {
    private static final String TAG = "PerformanceOptimization";

    // 解析器缓存
    private static final Map<Class<?>, Object> parserCache = new ConcurrentHashMap<>();

    // 对象池
    private static final Map<Class<?>, List<Object>> objectPool = new HashMap<>();

    // 自定义高效TypeAdapter
    public static class EfficientUserAdapter extends TypeAdapter<User> {
        @Override
        public void write(JsonWriter out, User value) throws IOException {
            out.beginObject();
            out.name("id").value(value.getId());
            out.name("name").value(value.getName());
            out.name("email").value(value.getEmail());
            out.name("active").value(value.isActive());
            out.endObject();
        }

        @Override
        public User read(JsonReader in) throws IOException {
            User user = getUserFromPool(User.class);
            user.reset();

            in.beginObject();
            while (in.hasNext()) {
                String name = in.nextName();
                switch (name) {
                    case "id":
                        user.setId(in.nextInt());
                        break;
                    case "name":
                        user.setName(in.nextString());
                        break;
                    case "email":
                        user.setEmail(in.nextString());
                        break;
                    case "active":
                        user.setActive(in.nextBoolean());
                        break;
                    default:
                        in.skipValue(); // 跳过未知字段
                        break;
                }
            }
            in.endObject();

            return user;
        }
    }

    // 用户数据模型（优化版）
    public static class User {
        private int id;
        private String name;
        private String email;
        private boolean active;

        public User() {}

        public User(int id, String name, String email, boolean active) {
            this.id = id;
            this.name = name;
            this.email = email;
            this.active = active;
        }

        // 重置对象状态（用于对象池）
        public void reset() {
            this.id = 0;
            this.name = null;
            this.email = null;
            this.active = false;
        }

        // Getters and Setters
        public int getId() { return id; }
        public void setId(int id) { this.id = id; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public boolean isActive() { return active; }
        public void setActive(boolean active) { this.active = active; }

        @Override
        public String toString() {
            return String.format("User{id=%d, name='%s', email='%s', active=%s}",
                id, name, email, active);
        }
    }

    // 获取优化的Gson实例
    public static Gson getOptimizedGson() {
        Gson gson = (Gson) parserCache.get(Gson.class);
        if (gson == null) {
            gson = new GsonBuilder()
                    .registerTypeAdapter(User.class, new EfficientUserAdapter())
                    .disableHtmlEscaping() // 禁用HTML转义
                    .create();
            parserCache.put(Gson.class, gson);
        }
        return gson;
    }

    // 从对象池获取对象
    @SuppressWarnings("unchecked")
    public static <T> T getUserFromPool(Class<T> clazz) {
        List<Object> pool = objectPool.get(clazz);
        if (pool != null && !pool.isEmpty()) {
            return (T) pool.remove(pool.size() - 1);
        }
        try {
            return clazz.newInstance();
        } catch (Exception e) {
            Log.e(TAG, "创建对象实例失败", e);
            return null;
        }
    }

    // 将对象返回到对象池
    public static <T> void returnUserToPool(T object) {
        if (object == null) return;

        Class<?> clazz = object.getClass();
        List<Object> pool = objectPool.computeIfAbsent(clazz, k -> new ArrayList<>());

        // 限制池大小，避免内存泄漏
        if (pool.size() < 100) {
            pool.add(object);
        }
    }

    // 批量解析优化
    public static List<User> parseUsersOptimized(String jsonArray) {
        long startTime = System.currentTimeMillis();
        List<User> users = new ArrayList<>();

        try {
            Gson gson = getOptimizedGson();
            User[] userArray = gson.fromJson(jsonArray, User[].class);

            for (User user : userArray) {
                users.add(user);
            }

            long endTime = System.currentTimeMillis();
            Log.d(TAG, String.format("批量解析完成，解析了%d个用户，耗时: %dms",
                users.size(), endTime - startTime));

        } catch (Exception e) {
            Log.e(TAG, "批量解析失败", e);
        }

        return users;
    }

    // 流式解析优化（适用于大型JSON数据）
    public static List<User> parseUsersStreaming(String jsonContent) {
        long startTime = System.currentTimeMillis();
        List<User> users = new ArrayList<>();

        try {
            Gson gson = getOptimizedGson();

            // 使用流式API逐个解析对象
            com.google.gson.stream.JsonReader reader = new com.google.gson.stream.JsonReader(
                new java.io.StringReader(jsonContent));

            reader.beginArray();
            while (reader.hasNext()) {
                User user = gson.fromJson(reader, User.class);
                if (user != null) {
                    users.add(user);
                }
            }
            reader.endArray();

            long endTime = System.currentTimeMillis();
            Log.d(TAG, String.format("流式解析完成，解析了%d个用户，耗时: %dms",
                users.size(), endTime - startTime));

        } catch (Exception e) {
            Log.e(TAG, "流式解析失败", e);
        }

        return users;
    }

    // 性能测试
    public static void performanceTest() {
        // 创建测试数据
        List<User> testUsers = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            testUsers.add(new User(i, "用户" + i, "user" + i + "@example.com", i % 2 == 0));
        }

        Gson gson = new GsonBuilder().create();
        String jsonData = gson.toJson(testUsers);
        System.out.println("测试数据大小: " + jsonData.length() + " 字符");

        // 1. 标准解析测试
        long startTime = System.currentTimeMillis();
        List<User> standardResult = gson.fromJson(jsonData, List.class);
        long standardTime = System.currentTimeMillis() - startTime;
        System.out.println("标准解析耗时: " + standardTime + "ms");

        // 2. 优化解析测试
        startTime = System.currentTimeMillis();
        List<User> optimizedResult = parseUsersOptimized(jsonData);
        long optimizedTime = System.currentTimeMillis() - startTime;
        System.out.println("优化解析耗时: " + optimizedTime + "ms");

        // 3. 流式解析测试
        startTime = System.currentTimeMillis();
        List<User> streamingResult = parseUsersStreaming(jsonData);
        long streamingTime = System.currentTimeMillis() - startTime;
        System.out.println("流式解析耗时: " + streamingTime + "ms");

        // 性能提升计算
        if (standardTime > 0) {
            double optimizedImprovement = (double) (standardTime - optimizedTime) / standardTime * 100;
            double streamingImprovement = (double) (standardTime - streamingTime) / standardTime * 100;

            System.out.println(String.format("优化解析性能提升: %.2f%%", optimizedImprovement));
            System.out.println(String.format("流式解析性能提升: %.2f%%", streamingImprovement));
        }
    }

    // 内存优化建议
    public static class MemoryOptimizationTips {

        // 1. 重用Gson实例
        private static final Gson SHARED_GSON = new GsonBuilder().create();

        // 2. 使用对象池
        public static <T> T getFromPool(Class<T> clazz) {
            return getUserFromPool(clazz);
        }

        // 3. 及时释放资源
        public static <T> void releaseToPool(T object) {
            returnUserToPool(object);
        }

        // 4. 避免深度嵌套
        public static String flattenJson(String json) {
            // 实现JSON扁平化逻辑
            return json; // 简化示例
        }

        // 5. 分批处理大数据
        public static void processInBatches(List<String> jsonList, int batchSize) {
            for (int i = 0; i < jsonList.size(); i += batchSize) {
                int endIndex = Math.min(i + batchSize, jsonList.size());
                List<String> batch = jsonList.subList(i, endIndex);
                processBatch(batch);
            }
        }

        private static void processBatch(List<String> batch) {
            // 处理批量数据
            for (String json : batch) {
                // 解析和处理每个JSON对象
            }
        }
    }

    public static void main(String[] args) {
        // 执行性能测试
        performanceTest();

        // 测试对象池
        User user1 = getUserFromPool(User.class);
        user1.setId(1);
        user1.setName("测试用户");
        System.out.println("从对象池获取: " + user1);

        returnUserToPool(user1);
        User user2 = getUserFromPool(User.class);
        System.out.println("重用对象池对象: " + user2);
    }
}
```

## 📱 实践示例：数据解析管理器

### 统一数据解析管理器

```java
// DataParseManager.java
import android.content.Context;
import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DataParseManager {
    private static final String TAG = "DataParseManager";
    private static DataParseManager instance;

    private final Context context;
    private final Gson gson;
    private final ExecutorService executorService;
    private final DataCache dataCache;

    private DataParseManager(Context context) {
        this.context = context.getApplicationContext();
        this.gson = new GsonBuilder()
                .setDateFormat("yyyy-MM-dd HH:mm:ss")
                .serializeNulls()
                .create();
        this.executorService = Executors.newFixedThreadPool(4);
        this.dataCache = new DataCache();
    }

    public static synchronized DataParseManager getInstance(Context context) {
        if (instance == null) {
            instance = new DataParseManager(context);
        }
        return instance;
    }

    // 同步解析JSON字符串
    public <T> T parseJson(String json, Class<T> clazz) {
        try {
            long startTime = System.currentTimeMillis();
            T result = gson.fromJson(json, clazz);
            long duration = System.currentTimeMillis() - startTime;

            Log.d(TAG, String.format("JSON解析完成，耗时: %dms", duration));
            return result;

        } catch (JsonSyntaxException e) {
            Log.e(TAG, "JSON语法错误", e);
            return null;
        } catch (Exception e) {
            Log.e(TAG, "JSON解析失败", e);
            return null;
        }
    }

    // 异步解析JSON字符串
    public <T> void parseJsonAsync(String json, Class<T> clazz, ParseCallback<T> callback) {
        executorService.execute(() -> {
            T result = parseJson(json, clazz);
            if (callback != null) {
                if (result != null) {
                    callback.onSuccess(result);
                } else {
                    callback.onError(new Exception("解析失败"));
                }
            }
        });
    }

    // 解析JSON数组
    public <T> List<T> parseJsonArray(String jsonArray, Class<T> clazz) {
        try {
            Type listType = com.google.gson.reflect.TypeToken.getParameterizedType(List.class, clazz);
            return gson.fromJson(jsonArray, listType);
        } catch (Exception e) {
            Log.e(TAG, "JSON数组解析失败", e);
            return null;
        }
    }

    // 解析JSON数组（异步）
    public <T> void parseJsonArrayAsync(String jsonArray, Class<T> clazz, ParseCallback<List<T>> callback) {
        executorService.execute(() -> {
            List<T> result = parseJsonArray(jsonArray, clazz);
            if (callback != null) {
                if (result != null) {
                    callback.onSuccess(result);
                } else {
                    callback.onError(new Exception("数组解析失败"));
                }
            }
        });
    }

    // 从Assets文件解析JSON
    public <T> T parseFromAssets(String fileName, Class<T> clazz) {
        try {
            String json = readFromAssets(fileName);
            if (json != null) {
                return parseJson(json, clazz);
            }
        } catch (Exception e) {
            Log.e(TAG, "从Assets解析JSON失败", e);
        }
        return null;
    }

    // 从Assets文件解析JSON（异步）
    public <T> void parseFromAssetsAsync(String fileName, Class<T> clazz, ParseCallback<T> callback) {
        executorService.execute(() -> {
            T result = parseFromAssets(fileName, clazz);
            if (callback != null) {
                if (result != null) {
                    callback.onSuccess(result);
                } else {
                    callback.onError(new Exception("Assets文件解析失败"));
                }
            }
        });
    }

    // 从Assets文件读取内容
    private String readFromAssets(String fileName) {
        try {
            InputStream inputStream = context.getAssets().open(fileName);
            int size = inputStream.available();
            byte[] buffer = new byte[size];
            inputStream.read(buffer);
            inputStream.close();
            return new String(buffer, "UTF-8");
        } catch (IOException e) {
            Log.e(TAG, "读取Assets文件失败: " + fileName, e);
            return null;
        }
    }

    // 序列化对象为JSON字符串
    public String toJson(Object object) {
        try {
            return gson.toJson(object);
        } catch (Exception e) {
            Log.e(TAG, "JSON序列化失败", e);
            return null;
        }
    }

    // 使用原生JSON解析器
    public <T> T parseWithNativeJson(String json, Class<T> clazz, NativeJsonParser<T> parser) {
        try {
            JSONObject jsonObject = new JSONObject(json);
            return parser.parse(jsonObject);
        } catch (JSONException e) {
            Log.e(TAG, "原生JSON解析失败", e);
            return null;
        }
    }

    // 使用原生JSON解析数组
    public <T> List<T> parseArrayWithNativeJson(String jsonArray, Class<T> clazz, NativeJsonParser<T> parser) {
        try {
            JSONArray array = new JSONArray(jsonArray);
            List<T> result = new java.util.ArrayList<>();

            for (int i = 0; i < array.length(); i++) {
                JSONObject jsonObject = array.getJSONObject(i);
                T item = parser.parse(jsonObject);
                if (item != null) {
                    result.add(item);
                }
            }

            return result;
        } catch (JSONException e) {
            Log.e(TAG, "原生JSON数组解析失败", e);
            return null;
        }
    }

    // 验证JSON格式
    public boolean isValidJson(String json) {
        try {
            new JSONObject(json);
            return true;
        } catch (JSONException e) {
            try {
                new JSONArray(json);
                return true;
            } catch (JSONException e2) {
                return false;
            }
        }
    }

    // 格式化JSON字符串
    public String formatJson(String json) {
        try {
            if (json.trim().startsWith("{")) {
                JSONObject jsonObject = new JSONObject(json);
                return jsonObject.toString(4);
            } else if (json.trim().startsWith("[")) {
                JSONArray jsonArray = new JSONArray(json);
                return jsonArray.toString(4);
            }
        } catch (JSONException e) {
            Log.e(TAG, "JSON格式化失败", e);
        }
        return json;
    }

    // 清理资源
    public void cleanup() {
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
        }
        if (dataCache != null) {
            dataCache.clear();
        }
    }

    // 解析回调接口
    public interface ParseCallback<T> {
        void onSuccess(T result);
        void onError(Exception error);
    }

    // 原生JSON解析器接口
    public interface NativeJsonParser<T> {
        T parse(JSONObject jsonObject) throws JSONException;
    }

    // 数据缓存类
    private static class DataCache {
        private final java.util.Map<String, Object> cache;
        private final java.util.Map<String, Long> timestamps;
        private static final long CACHE_DURATION = 5 * 60 * 1000; // 5分钟

        public DataCache() {
            this.cache = new java.util.concurrent.ConcurrentHashMap<>();
            this.timestamps = new java.util.concurrent.ConcurrentHashMap<>();
        }

        public void put(String key, Object value) {
            cache.put(key, value);
            timestamps.put(key, System.currentTimeMillis());
        }

        @SuppressWarnings("unchecked")
        public <T> T get(String key) {
            Long timestamp = timestamps.get(key);
            if (timestamp != null && (System.currentTimeMillis() - timestamp) < CACHE_DURATION) {
                return (T) cache.get(key);
            }
            remove(key);
            return null;
        }

        public void remove(String key) {
            cache.remove(key);
            timestamps.remove(key);
        }

        public void clear() {
            cache.clear();
            timestamps.clear();
        }
    }

    // 使用示例
    public static void main(String[] args) {
        // 模拟Android环境
        Context mockContext = null; // 在实际使用中传入真实的Context

        if (mockContext != null) {
            DataParseManager manager = DataParseManager.getInstance(mockContext);

            // 测试数据
            String jsonData = "{\"id\":1,\"name\":\"测试用户\",\"email\":\"test@example.com\"}";

            // 同步解析
            TestData data = manager.parseJson(jsonData, TestData.class);
            System.out.println("同步解析结果: " + data);

            // 异步解析
            manager.parseJsonAsync(jsonData, TestData.class, new ParseCallback<TestData>() {
                @Override
                public void onSuccess(TestData result) {
                    System.out.println("异步解析成功: " + result);
                }

                @Override
                public void onError(Exception error) {
                    System.out.println("异步解析失败: " + error.getMessage());
                }
            });

            // 原生JSON解析
            manager.parseWithNativeJson(jsonData, TestData.class, new NativeJsonParser<TestData>() {
                @Override
                public TestData parse(JSONObject jsonObject) throws JSONException {
                    TestData data = new TestData();
                    data.setId(jsonObject.getInt("id"));
                    data.setName(jsonObject.getString("name"));
                    data.setEmail(jsonObject.optString("email"));
                    return data;
                }
            });
        }
    }

    // 测试数据类
    public static class TestData {
        private int id;
        private String name;
        private String email;

        public TestData() {}

        public TestData(int id, String name, String email) {
            this.id = id;
            this.name = name;
            this.email = email;
        }

        // Getters and Setters
        public int getId() { return id; }
        public void setId(int id) { this.id = id; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }

        @Override
        public String toString() {
            return String.format("TestData{id=%d, name='%s', email='%s'}", id, name, email);
        }
    }
}
```

## 📝 本章小结

### 核心知识点

1. **Gson库使用详解**
   - 基本序列化和反序列化
   - 高级配置和自定义适配器
   - 注解使用和字段映射
   - 日期和复杂数据类型处理

2. **Moshi高级特性**
   - 现代化的JSON解析方式
   - 自定义TypeAdapter
   - 枚举和特殊类型处理
   - 类型安全的数据绑定

3. **XML解析技术**
   - DOM解析器实现和使用
   - SAX解析器事件驱动机制
   - XML安全配置和防护
   - 性能对比和选择建议

4. **数据绑定和转换**
   - 统一解析接口设计
   - 多种解析器支持
   - 数据转换工具类
   - 对象池和缓存优化

5. **性能优化技巧**
   - 自定义TypeAdapter优化
   - 对象池管理
   - 批量处理和流式解析
   - 内存优化策略

6. **解析管理器**
   - 统一数据解析管理
   - 异步解析支持
   - 原生JSON解析器
   - 数据验证和格式化

### 实践建议

1. **解析器选择**
   - Gson：功能全面，易于使用，适合大多数场景
   - Moshi：性能更好，类型安全，适合高性能要求的应用
   - DOM XML：适合小型XML文件，便于随机访问
   - SAX XML：适合大型XML文件，内存占用低

2. **性能优化**
   - 重用解析器实例
   - 使用对象池减少GC压力
   - 批量处理大数据
   - 实现自定义适配器

3. **错误处理**
   - 提供详细的错误信息
   - 实现重试机制
   - 使用异步解析避免阻塞UI线程
   - 添加数据验证

4. **安全考虑**
   - 防止XML外部实体攻击
   - 验证输入数据格式
   - 限制解析数据大小
   - 使用安全的解析器配置

### 常见问题解决

1. **JSON解析错误**
   - 检查JSON格式是否正确
   - 确认字段名称和类型匹配
   - 使用try-catch处理解析异常
   - 添加数据验证逻辑

2. **性能问题**
   - 优化数据结构设计
   - 使用流式解析处理大数据
   - 实现对象池和缓存
   - 避免深度嵌套结构

3. **内存泄漏**
   - 及时释放资源
   - 限制对象池大小
   - 清理缓存数据
   - 使用弱引用缓存

4. **类型转换问题**
   - 实现自定义类型适配器
   - 使用注解映射字段
   - 处理null值和默认值
   - 提供类型转换工具

通过本章的学习，你掌握了Android中JSON和XML数据解析的完整技术栈，包括Gson和Moshi的使用、DOM和SAX XML解析、数据绑定转换、性能优化和统一解析管理。这些技能为处理各种网络数据和本地数据提供了强大的支持。