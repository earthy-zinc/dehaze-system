# 第20章：HTTP请求和响应处理

## 🔄 HttpURLConnection使用

### HttpURLConnection基础用法

HttpURLConnection是Android系统提供的原生HTTP客户端，虽然相对复杂，但在某些场景下仍然很有用。

#### 基本GET请求

```java
// HttpURLConnectionHelper.java
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

public class HttpURLConnectionHelper {
    private static final String TAG = "HttpURLConnectionHelper";
    private static final int DEFAULT_TIMEOUT = 30000; // 30秒超时

    // GET请求
    public static String get(String urlString) throws IOException {
        return get(urlString, new HashMap<>(), DEFAULT_TIMEOUT);
    }

    // 带请求头的GET请求
    public static String get(String urlString, Map<String, String> headers, int timeout) throws IOException {
        HttpURLConnection connection = null;
        InputStream inputStream = null;
        BufferedReader reader = null;

        try {
            URL url = new URL(urlString);
            connection = (HttpURLConnection) url.openConnection();

            // 设置请求方法和超时
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(timeout);
            connection.setReadTimeout(timeout);

            // 设置请求头
            for (Map.Entry<String, String> entry : headers.entrySet()) {
                connection.setRequestProperty(entry.getKey(), entry.getValue());
            }

            // 设置通用请求头
            connection.setRequestProperty("User-Agent", "AndroidApp/1.0");
            connection.setRequestProperty("Accept", "application/json");
            connection.setRequestProperty("Accept-Encoding", "gzip, deflate");

            // 启用GZIP压缩
            connection.setRequestProperty("Accept-Encoding", "gzip");

            // 连接
            connection.connect();

            // 检查响应码
            int responseCode = connection.getResponseCode();
            if (responseCode != HttpURLConnection.HTTP_OK) {
                throw new IOException("HTTP错误: " + responseCode);
            }

            // 读取响应
            inputStream = getInputStream(connection);
            reader = new BufferedReader(new InputStreamReader(inputStream));

            StringBuilder response = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                response.append(line);
            }

            Log.d(TAG, String.format("GET请求成功: %s, 响应大小: %d", urlString, response.length()));
            return response.toString();

        } finally {
            // 关闭资源
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭BufferedReader失败", e);
                }
            }
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭InputStream失败", e);
                }
            }
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    // 获取输入流（处理GZIP压缩）
    private static InputStream getInputStream(HttpURLConnection connection) throws IOException {
        String contentEncoding = connection.getContentEncoding();
        InputStream inputStream = connection.getInputStream();

        if ("gzip".equalsIgnoreCase(contentEncoding)) {
            return new java.util.zip.GZIPInputStream(inputStream);
        }

        return inputStream;
    }

    // POST请求
    public static String post(String urlString, String postData) throws IOException {
        return post(urlString, postData, "application/json", new HashMap<>(), DEFAULT_TIMEOUT);
    }

    // 带请求头的POST请求
    public static String post(String urlString, String postData, String contentType,
                              Map<String, String> headers, int timeout) throws IOException {
        HttpURLConnection connection = null;
        InputStream inputStream = null;
        BufferedReader reader = null;

        try {
            URL url = new URL(urlString);
            connection = (HttpURLConnection) url.openConnection();

            // 设置请求方法和超时
            connection.setRequestMethod("POST");
            connection.setConnectTimeout(timeout);
            connection.setReadTimeout(timeout);

            // 设置请求头
            for (Map.Entry<String, String> entry : headers.entrySet()) {
                connection.setRequestProperty(entry.getKey(), entry.getValue());
            }

            connection.setRequestProperty("User-Agent", "AndroidApp/1.0");
            connection.setRequestProperty("Content-Type", contentType);
            connection.setRequestProperty("Accept", "application/json");

            // 启用输出
            connection.setDoOutput(true);

            // 写入请求数据
            if (postData != null && !postData.isEmpty()) {
                byte[] postBytes = postData.getBytes("UTF-8");
                connection.setRequestProperty("Content-Length", String.valueOf(postBytes.length));

                try (java.io.OutputStream outputStream = connection.getOutputStream()) {
                    outputStream.write(postBytes);
                    outputStream.flush();
                }
            }

            // 连接
            connection.connect();

            // 检查响应码
            int responseCode = connection.getResponseCode();
            if (responseCode != HttpURLConnection.HTTP_OK) {
                throw new IOException("HTTP错误: " + responseCode);
            }

            // 读取响应
            inputStream = getInputStream(connection);
            reader = new BufferedReader(new InputStreamReader(inputStream));

            StringBuilder response = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                response.append(line);
            }

            Log.d(TAG, String.format("POST请求成功: %s, 响应大小: %d", urlString, response.length()));
            return response.toString();

        } finally {
            // 关闭资源
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭BufferedReader失败", e);
                }
            }
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭InputStream失败", e);
                }
            }
            if (connection != null) {
                connection.disconnect();
            }
        }
    }
}
```

### HttpURLConnection高级用法

```java
// AdvancedHttpURLConnectionHelper.java
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

public class AdvancedHttpURLConnectionHelper {
    private static final String TAG = "AdvancedHttpURLConnection";

    // 文件上传
    public static String uploadFile(String urlString, File file, String fileParamName,
                                   Map<String, String> formData) throws IOException {
        String boundary = "----" + System.currentTimeMillis();
        String lineEnd = "\r\n";
        String twoHyphens = "--";

        HttpURLConnection connection = null;
        OutputStream outputStream = null;
        FileInputStream fileInputStream = null;

        try {
            URL url = new URL(urlString);
            connection = (HttpURLConnection) url.openConnection();

            // 设置请求方法和超时
            connection.setRequestMethod("POST");
            connection.setConnectTimeout(30000);
            connection.setReadTimeout(30000);

            // 设置请求头
            connection.setRequestProperty("User-Agent", "AndroidApp/1.0");
            connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

            // 启用输出
            connection.setDoOutput(true);
            connection.setDoInput(true);

            // 写入表单数据
            outputStream = connection.getOutputStream();

            // 写入普通表单字段
            if (formData != null) {
                for (Map.Entry<String, String> entry : formData.entrySet()) {
                    outputStream.write((twoHyphens + boundary + lineEnd).getBytes(StandardCharsets.UTF_8));
                    outputStream.write(("Content-Disposition: form-data; name=\"" + entry.getKey() + "\"" + lineEnd).getBytes(StandardCharsets.UTF_8));
                    outputStream.write((lineEnd).getBytes(StandardCharsets.UTF_8));
                    outputStream.write((entry.getValue() + lineEnd).getBytes(StandardCharsets.UTF_8));
                }
            }

            // 写入文件数据
            outputStream.write((twoHyphens + boundary + lineEnd).getBytes(StandardCharsets.UTF_8));
            outputStream.write(("Content-Disposition: form-data; name=\"" + fileParamName +
                "\"; filename=\"" + file.getName() + "\"" + lineEnd).getBytes(StandardCharsets.UTF_8));
            outputStream.write(("Content-Type: " + getContentType(file.getName()) + lineEnd).getBytes(StandardCharsets.UTF_8));
            outputStream.write((lineEnd).getBytes(StandardCharsets.UTF_8));

            // 写入文件内容
            byte[] buffer = new byte[8192];
            int bytesRead;
            fileInputStream = new FileInputStream(file);
            while ((bytesRead = fileInputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }

            // 结束边界
            outputStream.write((lineEnd + twoHyphens + boundary + twoHyphens + lineEnd).getBytes(StandardCharsets.UTF_8));
            outputStream.flush();

            // 获取响应
            int responseCode = connection.getResponseCode();
            if (responseCode != HttpURLConnection.HTTP_OK) {
                throw new IOException("上传失败，HTTP错误: " + responseCode);
            }

            // 读取响应
            return readResponse(connection);

        } finally {
            // 关闭资源
            if (fileInputStream != null) {
                try {
                    fileInputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭FileInputStream失败", e);
                }
            }
            if (outputStream != null) {
                try {
                    outputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭OutputStream失败", e);
                }
            }
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    // 根据文件扩展名获取Content-Type
    private static String getContentType(String fileName) {
        String extension = fileName.substring(fileName.lastIndexOf('.') + 1).toLowerCase();
        switch (extension) {
            case "jpg":
            case "jpeg":
                return "image/jpeg";
            case "png":
                return "image/png";
            case "gif":
                return "image/gif";
            case "pdf":
                return "application/pdf";
            case "txt":
                return "text/plain";
            case "json":
                return "application/json";
            default:
                return "application/octet-stream";
        }
    }

    // 读取响应
    private static String readResponse(HttpURLConnection connection) throws IOException {
        InputStream inputStream = null;
        ByteArrayOutputStream byteArrayOutputStream = null;

        try {
            inputStream = connection.getInputStream();
            byteArrayOutputStream = new ByteArrayOutputStream();

            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                byteArrayOutputStream.write(buffer, 0, bytesRead);
            }

            return byteArrayOutputStream.toString("UTF-8");

        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭InputStream失败", e);
                }
            }
            if (byteArrayOutputStream != null) {
                try {
                    byteArrayOutputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭ByteArrayOutputStream失败", e);
                }
            }
        }
    }

    // URL编码
    public static String encode(String text) {
        try {
            return URLEncoder.encode(text, "UTF-8");
        } catch (Exception e) {
            Log.e(TAG, "URL编码失败", e);
            return text;
        }
    }

    // 构建查询字符串
    public static String buildQueryString(Map<String, String> params) {
        if (params == null || params.isEmpty()) {
            return "";
        }

        StringBuilder query = new StringBuilder();
        for (Map.Entry<String, String> entry : params.entrySet()) {
            if (query.length() > 0) {
                query.append("&");
            }
            query.append(encode(entry.getKey()))
                 .append("=")
                 .append(encode(entry.getValue()));
        }

        return query.toString();
    }
}
```

## ⚙️ 连接池和超时处理

### 连接池管理

```java
// ConnectionPoolManager.java
import android.util.Log;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import javax.net.ssl.HttpsURLConnection;

public class ConnectionPoolManager {
    private static final String TAG = "ConnectionPoolManager";
    private static ConnectionPoolManager instance;

    private final ExecutorService executorService;
    private final AtomicInteger activeConnections;
    private final ConnectionPool connectionPool;
    private final int maxConnections;
    private final long keepAliveTime;

    private ConnectionPoolManager() {
        this.maxConnections = 20; // 最大连接数
        this.keepAliveTime = 300000; // 5分钟保活时间
        this.activeConnections = new AtomicInteger(0);
        this.executorService = Executors.newFixedThreadPool(maxConnections);
        this.connectionPool = new ConnectionPool(maxConnections, keepAliveTime, TimeUnit.MILLISECONDS);

        // 配置JVM级别的HTTP连接设置
        System.setProperty("http.keepAlive", "true");
        System.setProperty("http.maxConnections", String.valueOf(maxConnections));
        System.setProperty("https.maxConnections", String.valueOf(maxConnections));
    }

    public static synchronized ConnectionPoolManager getInstance() {
        if (instance == null) {
            instance = new ConnectionPoolManager();
        }
        return instance;
    }

    // 执行HTTP请求
    public void executeRequest(RequestTask task) {
        if (activeConnections.get() >= maxConnections) {
            task.onError(new IOException("连接池已满"));
            return;
        }

        executorService.execute(() -> {
            activeConnections.incrementAndGet();
            try {
                task.run();
            } finally {
                activeConnections.decrementAndGet();
            }
        });
    }

    // 获取连接池状态
    public PoolStatus getPoolStatus() {
        return new PoolStatus(
            activeConnections.get(),
            maxConnections,
            connectionPool.connectionCount(),
            connectionPool.idleConnectionCount()
        );
    }

    // 关闭连接池
    public void shutdown() {
        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(10, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
            Thread.currentThread().interrupt();
        }
        connectionPool.evictAll();
    }

    // 连接池状态
    public static class PoolStatus {
        private final int activeConnections;
        private final int maxConnections;
        private final int totalConnections;
        private final int idleConnections;

        public PoolStatus(int activeConnections, int maxConnections, int totalConnections, int idleConnections) {
            this.activeConnections = activeConnections;
            this.maxConnections = maxConnections;
            this.totalConnections = totalConnections;
            this.idleConnections = idleConnections;
        }

        // Getters
        public int getActiveConnections() { return activeConnections; }
        public int getMaxConnections() { return maxConnections; }
        public int getTotalConnections() { return totalConnections; }
        public int getIdleConnections() { return idleConnections; }

        @Override
        public String toString() {
            return String.format("PoolStatus{active=%d, max=%d, total=%d, idle=%d}",
                activeConnections, maxConnections, totalConnections, idleConnections);
        }
    }

    // 请求任务接口
    public interface RequestTask {
        void run();
        void onError(Exception error);
    }
}
```

### 超时处理策略

```java
// TimeoutManager.java
import android.util.Log;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public class TimeoutManager {
    private static final String TAG = "TimeoutManager";
    private final ExecutorService timeoutExecutor;

    public TimeoutManager() {
        this.timeoutExecutor = Executors.newCachedThreadPool();
    }

    // 执行带超时的HTTP请求
    public <T> T executeWithTimeout(Callable<T> task, long timeout, TimeUnit unit)
        throws IOException, TimeoutException {

        try {
            Future<T> future = timeoutExecutor.submit(task);
            return future.get(timeout, unit);
        } catch (TimeoutException e) {
            Log.w(TAG, "请求超时: " + timeout + " " + unit);
            throw new TimeoutException("请求超时: " + timeout + " " + unit);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("请求被中断", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof IOException) {
                throw (IOException) cause;
            } else if (cause instanceof SocketTimeoutException) {
                throw new TimeoutException("连接超时: " + cause.getMessage());
            } else {
                throw new IOException("请求执行失败", cause);
            }
        }
    }

    // 带重试的请求
    public <T> T executeWithRetry(Callable<T> task, int maxRetries, long initialDelay,
                                  double backoffMultiplier) throws IOException {

        int retryCount = 0;
        IOException lastException = null;
        long delay = initialDelay;

        while (retryCount <= maxRetries) {
            try {
                return task.call();
            } catch (SocketTimeoutException e) {
                lastException = new IOException("连接超时", e);
                Log.w(TAG, String.format("请求超时，重试 %d/%d", retryCount, maxRetries));
            } catch (IOException e) {
                lastException = e;
                Log.w(TAG, String.format("请求失败，重试 %d/%d: %s", retryCount, maxRetries, e.getMessage()));
            } catch (Exception e) {
                throw new IOException("请求执行失败", e);
            }

            if (retryCount < maxRetries) {
                try {
                    Thread.sleep(delay);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new IOException("重试被中断", ie);
                }
                delay = (long) (delay * backoffMultiplier);
            }

            retryCount++;
        }

        throw lastException != null ? lastException : new IOException("重试次数已用尽");
    }

    // 配置HttpURLConnection超时
    public static void configureTimeouts(HttpURLConnection connection, int connectTimeout,
                                        int readTimeout) {
        connection.setConnectTimeout(connectTimeout);
        connection.setReadTimeout(readTimeout);

        Log.d(TAG, String.format("配置超时: connect=%dms, read=%dms", connectTimeout, readTimeout));
    }

    // 超时配置类
    public static class TimeoutConfig {
        private final int connectTimeout;
        private final int readTimeout;
        private final int totalTimeout;
        private final int maxRetries;
        private final long retryDelay;
        private final double backoffMultiplier;

        public TimeoutConfig(int connectTimeout, int readTimeout, int totalTimeout,
                           int maxRetries, long retryDelay, double backoffMultiplier) {
            this.connectTimeout = connectTimeout;
            this.readTimeout = readTimeout;
            this.totalTimeout = totalTimeout;
            this.maxRetries = maxRetries;
            this.retryDelay = retryDelay;
            this.backoffMultiplier = backoffMultiplier;
        }

        // 预定义配置
        public static TimeoutConfig getFastConfig() {
            return new TimeoutConfig(5000, 10000, 15000, 2, 1000, 1.5);
        }

        public static TimeoutConfig getNormalConfig() {
            return new TimeoutConfig(10000, 30000, 60000, 3, 2000, 2.0);
        }

        public static TimeoutConfig getSlowConfig() {
            return new TimeoutConfig(15000, 60000, 120000, 5, 5000, 2.0);
        }

        // Getters
        public int getConnectTimeout() { return connectTimeout; }
        public int getReadTimeout() { return readTimeout; }
        public int getTotalTimeout() { return totalTimeout; }
        public int getMaxRetries() { return maxRetries; }
        public long getRetryDelay() { return retryDelay; }
        public double getBackoffMultiplier() { return backoffMultiplier; }
    }

    // 关闭超时管理器
    public void shutdown() {
        timeoutExecutor.shutdown();
        try {
            if (!timeoutExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                timeoutExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            timeoutExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
```

## 🛡️ 请求和响应拦截

### 拦截器模式实现

```java
// HttpInterceptor.java
import android.util.Log;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class HttpInterceptor {
    private static final String TAG = "HttpInterceptor";
    private final List<RequestInterceptor> requestInterceptors;
    private final List<ResponseInterceptor> responseInterceptors;

    public HttpInterceptor() {
        this.requestInterceptors = new ArrayList<>();
        this.responseInterceptors = new ArrayList<>();
    }

    // 添加请求拦截器
    public void addRequestInterceptor(RequestInterceptor interceptor) {
        requestInterceptors.add(interceptor);
    }

    // 添加响应拦截器
    public void addResponseInterceptor(ResponseInterceptor interceptor) {
        responseInterceptors.add(interceptor);
    }

    // 执行请求拦截
    public RequestData interceptRequest(RequestData request) throws IOException {
        RequestData result = request;
        for (RequestInterceptor interceptor : requestInterceptors) {
            result = interceptor.intercept(result);
        }
        return result;
    }

    // 执行响应拦截
    public ResponseData interceptResponse(ResponseData response) throws IOException {
        ResponseData result = response;
        for (ResponseInterceptor interceptor : responseInterceptors) {
            result = interceptor.intercept(result);
        }
        return result;
    }

    // 请求数据封装
    public static class RequestData {
        private String url;
        private String method;
        private Map<String, String> headers;
        private byte[] body;

        public RequestData(String url, String method, Map<String, String> headers, byte[] body) {
            this.url = url;
            this.method = method;
            this.headers = headers != null ? new HashMap<>(headers) : new HashMap<>();
            this.body = body;
        }

        // Getters and Setters
        public String getUrl() { return url; }
        public void setUrl(String url) { this.url = url; }
        public String getMethod() { return method; }
        public void setMethod(String method) { this.method = method; }
        public Map<String, String> getHeaders() { return headers; }
        public void setHeaders(Map<String, String> headers) { this.headers = headers; }
        public byte[] getBody() { return body; }
        public void setBody(byte[] body) { this.body = body; }

        public void addHeader(String name, String value) {
            headers.put(name, value);
        }

        public String getHeader(String name) {
            return headers.get(name);
        }
    }

    // 响应数据封装
    public static class ResponseData {
        private int statusCode;
        private Map<String, String> headers;
        private byte[] body;

        public ResponseData(int statusCode, Map<String, String> headers, byte[] body) {
            this.statusCode = statusCode;
            this.headers = headers != null ? new HashMap<>(headers) : new HashMap<>();
            this.body = body;
        }

        // Getters and Setters
        public int getStatusCode() { return statusCode; }
        public void setStatusCode(int statusCode) { this.statusCode = statusCode; }
        public Map<String, String> getHeaders() { return headers; }
        public void setHeaders(Map<String, String> headers) { this.headers = headers; }
        public byte[] getBody() { return body; }
        public void setBody(byte[] body) { this.body = body; }

        public void addHeader(String name, String value) {
            headers.put(name, value);
        }

        public String getHeader(String name) {
            return headers.get(name);
        }

        public boolean isSuccessful() {
            return statusCode >= 200 && statusCode < 300;
        }
    }

    // 请求拦截器接口
    public interface RequestInterceptor {
        RequestData intercept(RequestData request) throws IOException;
    }

    // 响应拦截器接口
    public interface ResponseInterceptor {
        ResponseData intercept(ResponseData response) throws IOException;
    }

    // 预定义拦截器
    public static class UserAgentInterceptor implements RequestInterceptor {
        private final String userAgent;

        public UserAgentInterceptor(String userAgent) {
            this.userAgent = userAgent;
        }

        @Override
        public RequestData intercept(RequestData request) throws IOException {
            request.addHeader("User-Agent", userAgent);
            Log.d(TAG, "添加User-Agent: " + userAgent);
            return request;
        }
    }

    public static class AuthInterceptor implements RequestInterceptor {
        private final String token;

        public AuthInterceptor(String token) {
            this.token = token;
        }

        @Override
        public RequestData intercept(RequestData request) throws IOException {
            if (token != null && !token.isEmpty()) {
                request.addHeader("Authorization", "Bearer " + token);
                Log.d(TAG, "添加Authorization头");
            }
            return request;
        }
    }

    public static class LoggingInterceptor implements RequestInterceptor, ResponseInterceptor {
        private final boolean logHeaders;
        private final boolean logBody;

        public LoggingInterceptor(boolean logHeaders, boolean logBody) {
            this.logHeaders = logHeaders;
            this.logBody = logBody;
        }

        @Override
        public RequestData intercept(RequestData request) throws IOException {
            logRequest(request);
            return request;
        }

        @Override
        public ResponseData intercept(ResponseData response) throws IOException {
            logResponse(response);
            return response;
        }

        private void logRequest(RequestData request) {
            Log.d(TAG, String.format("请求: %s %s", request.getMethod(), request.getUrl()));

            if (logHeaders) {
                Log.d(TAG, "请求头:");
                for (Map.Entry<String, String> entry : request.getHeaders().entrySet()) {
                    Log.d(TAG, String.format("  %s: %s", entry.getKey(), entry.getValue()));
                }
            }

            if (logBody && request.getBody() != null) {
                Log.d(TAG, String.format("请求体大小: %d bytes", request.getBody().length));
            }
        }

        private void logResponse(ResponseData response) {
            Log.d(TAG, String.format("响应: %d", response.getStatusCode()));

            if (logHeaders) {
                Log.d(TAG, "响应头:");
                for (Map.Entry<String, String> entry : response.getHeaders().entrySet()) {
                    Log.d(TAG, String.format("  %s: %s", entry.getKey(), entry.getValue()));
                }
            }

            if (logBody && response.getBody() != null) {
                Log.d(TAG, String.format("响应体大小: %d bytes", response.getBody().length));
            }
        }
    }

    public static class CacheInterceptor implements ResponseInterceptor {
        private final Map<String, ResponseData> cache;

        public CacheInterceptor() {
            this.cache = new HashMap<>();
        }

        @Override
        public ResponseData intercept(ResponseData response) throws IOException {
            // 简单缓存实现
            if (response.getStatusCode() == HttpURLConnection.HTTP_OK) {
                String cacheKey = generateCacheKey(response);
                cache.put(cacheKey, response);
                Log.d(TAG, "缓存响应: " + cacheKey);
            }
            return response;
        }

        private String generateCacheKey(ResponseData response) {
            return System.currentTimeMillis() + "_" + response.getStatusCode();
        }

        public ResponseData getCachedResponse(String key) {
            return cache.get(key);
        }

        public void clearCache() {
            cache.clear();
            Log.d(TAG, "缓存已清除");
        }
    }
}
```

## 📁 文件上传下载

### 文件下载实现

```java
// FileDownloader.java
import android.app.DownloadManager;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.atomic.AtomicLong;

public class FileDownloader {
    private static final String TAG = "FileDownloader";
    private static final int BUFFER_SIZE = 8192;

    // 下载进度监听器
    public interface DownloadListener {
        void onProgress(long downloaded, long total);
        void onSuccess(File file);
        void onError(Exception error);
    }

    // 下载文件
    public static void downloadFile(String url, File outputFile, DownloadListener listener) {
        new Thread(() -> {
            try {
                downloadFileInternal(url, outputFile, listener);
            } catch (Exception e) {
                if (listener != null) {
                    listener.onError(e);
                }
            }
        }).start();
    }

    // 内部下载实现
    private static void downloadFileInternal(String url, File outputFile, DownloadListener listener)
        throws IOException {

        HttpURLConnection connection = null;
        InputStream inputStream = null;
        FileOutputStream outputStream = null;

        try {
            URL downloadUrl = new URL(url);
            connection = (HttpURLConnection) downloadUrl.openConnection();

            // 配置连接
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(30000);
            connection.setReadTimeout(30000);

            // 设置请求头
            connection.setRequestProperty("User-Agent", "AndroidApp/1.0");
            connection.setRequestProperty("Accept-Encoding", "gzip");

            // 支持断点续传
            if (outputFile.exists()) {
                long existingLength = outputFile.length();
                connection.setRequestProperty("Range", "bytes=" + existingLength + "-");
            }

            // 连接
            connection.connect();

            // 检查响应码
            int responseCode = connection.getResponseCode();
            if (responseCode != HttpURLConnection.HTTP_OK &&
                responseCode != HttpURLConnection.HTTP_PARTIAL) {
                throw new IOException("下载失败，HTTP错误: " + responseCode);
            }

            // 获取文件大小
            long contentLength = connection.getContentLength();
            if (contentLength <= 0) {
                contentLength = connection.getHeaderField("Content-Length") != null ?
                    Long.parseLong(connection.getHeaderField("Content-Length")) : -1;
            }

            // 处理断点续传
            long existingLength = outputFile.exists() ? outputFile.length() : 0;
            if (responseCode == HttpURLConnection.HTTP_PARTIAL) {
                Log.d(TAG, "继续下载，已下载: " + existingLength + " bytes");
            }

            // 创建输出流
            outputStream = new FileOutputStream(outputFile, existingLength > 0);
            inputStream = connection.getInputStream();

            // 下载文件
            byte[] buffer = new byte[BUFFER_SIZE];
            int bytesRead;
            AtomicLong totalBytesRead = new AtomicLong(existingLength);

            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
                totalBytesRead.addAndGet(bytesRead);

                // 通知进度
                if (listener != null) {
                    listener.onProgress(totalBytesRead.get(), contentLength);
                }
            }

            outputStream.flush();

            // 下载完成
            if (listener != null) {
                listener.onSuccess(outputFile);
            }

            Log.d(TAG, String.format("文件下载完成: %s, 大小: %d bytes",
                outputFile.getAbsolutePath(), totalBytesRead.get()));

        } finally {
            // 关闭资源
            if (outputStream != null) {
                try {
                    outputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭FileOutputStream失败", e);
                }
            }
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭InputStream失败", e);
                }
            }
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    // 使用DownloadManager下载
    public static long downloadWithManager(Context context, String url, String title, String description) {
        DownloadManager.Request request = new DownloadManager.Request(Uri.parse(url));

        // 设置下载属性
        request.setTitle(title);
        request.setDescription(description);
        request.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, title);

        // 设置网络类型
        request.setAllowedNetworkTypes(DownloadManager.Request.NETWORK_WIFI |
                                     DownloadManager.Request.NETWORK_MOBILE);

        // 设置通知
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.HONEYCOMB) {
            request.setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
        }

        // 设置允许漫游
        request.setAllowedOverRoaming(false);

        // 开始下载
        DownloadManager downloadManager = (DownloadManager) context.getSystemService(Context.DOWNLOAD_SERVICE);
        return downloadManager.enqueue(request);
    }

    // 查询下载状态
    public static DownloadStatus getDownloadStatus(Context context, long downloadId) {
        DownloadManager downloadManager = (DownloadManager) context.getSystemService(Context.DOWNLOAD_SERVICE);
        DownloadManager.Query query = new DownloadManager.Query();
        query.setFilterById(downloadId);

        Cursor cursor = null;
        try {
            cursor = downloadManager.query(query);
            if (cursor != null && cursor.moveToFirst()) {
                int status = cursor.getInt(cursor.getColumnIndex(DownloadManager.COLUMN_STATUS));
                int reason = cursor.getInt(cursor.getColumnIndex(DownloadManager.COLUMN_REASON));

                switch (status) {
                    case DownloadManager.STATUS_SUCCESSFUL:
                        return DownloadStatus.SUCCESS;
                    case DownloadManager.STATUS_FAILED:
                        return DownloadStatus.FAILED;
                    case DownloadManager.STATUS_PAUSED:
                        return DownloadStatus.PAUSED;
                    case DownloadManager.STATUS_PENDING:
                        return DownloadStatus.PENDING;
                    case DownloadManager.STATUS_RUNNING:
                        return DownloadStatus.RUNNING;
                    default:
                        return DownloadStatus.UNKNOWN;
                }
            }
        } finally {
            if (cursor != null) {
                cursor.close();
            }
        }

        return DownloadStatus.UNKNOWN;
    }

    // 下载状态枚举
    public enum DownloadStatus {
        SUCCESS,
        FAILED,
        PAUSED,
        PENDING,
        RUNNING,
        UNKNOWN
    }

    // 下载完成广播接收器
    public static class DownloadCompleteReceiver extends BroadcastReceiver {
        private final DownloadCompleteListener listener;

        public DownloadCompleteReceiver(DownloadCompleteListener listener) {
            this.listener = listener;
        }

        @Override
        public void onReceive(Context context, Intent intent) {
            long downloadId = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1);
            if (listener != null) {
                listener.onDownloadComplete(downloadId);
            }
        }

        public interface DownloadCompleteListener {
            void onDownloadComplete(long downloadId);
        }
    }
}
```

### 文件上传实现

```java
// FileUploader.java
import android.util.Log;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.atomic.AtomicLong;

public class FileUploader {
    private static final String TAG = "FileUploader";
    private static final int BUFFER_SIZE = 8192;
    private static final String LINE_END = "\r\n";
    private static final String TWO_HYPHENS = "--";

    // 上传进度监听器
    public interface UploadListener {
        void onProgress(long uploaded, long total);
        void onSuccess(String response);
        void onError(Exception error);
    }

    // 上传单个文件
    public static void uploadFile(String url, File file, String fileParamName,
                                 String fileName, UploadListener listener) {
        new Thread(() -> {
            try {
                uploadFileInternal(url, file, fileParamName, fileName, null, listener);
            } catch (Exception e) {
                if (listener != null) {
                    listener.onError(e);
                }
            }
        }).start();
    }

    // 上传多个文件
    public static void uploadFiles(String url, File[] files, String[] fileParamNames,
                                  UploadListener listener) {
        new Thread(() -> {
            try {
                uploadMultipleFilesInternal(url, files, fileParamNames, listener);
            } catch (Exception e) {
                if (listener != null) {
                    listener.onError(e);
                }
            }
        }).start();
    }

    // 内部上传实现
    private static void uploadFileInternal(String url, File file, String fileParamName,
                                         String fileName, String contentType, UploadListener listener)
        throws IOException {

        String boundary = "----" + System.currentTimeMillis();
        HttpURLConnection connection = null;
        OutputStream outputStream = null;
        FileInputStream fileInputStream = null;
        DataOutputStream dataOutputStream = null;

        try {
            URL uploadUrl = new URL(url);
            connection = (HttpURLConnection) uploadUrl.openConnection();

            // 配置连接
            connection.setRequestMethod("POST");
            connection.setConnectTimeout(30000);
            connection.setReadTimeout(30000);
            connection.setDoInput(true);
            connection.setDoOutput(true);
            connection.setUseCaches(false);

            // 设置请求头
            connection.setRequestProperty("Connection", "Keep-Alive");
            connection.setRequestProperty("User-Agent", "AndroidApp/1.0");
            connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

            // 获取输出流
            outputStream = connection.getOutputStream();
            dataOutputStream = new DataOutputStream(outputStream);

            // 写入文件数据
            dataOutputStream.writeBytes(TWO_HYPHENS + boundary + LINE_END);
            dataOutputStream.writeBytes("Content-Disposition: form-data; name=\"" +
                fileParamName + "\"; filename=\"" + fileName + "\"" + LINE_END);

            if (contentType != null) {
                dataOutputStream.writeBytes("Content-Type: " + contentType + LINE_END);
            } else {
                dataOutputStream.writeBytes("Content-Type: application/octet-stream" + LINE_END);
            }

            dataOutputStream.writeBytes(LINE_END);

            // 写入文件内容
            fileInputStream = new FileInputStream(file);
            byte[] buffer = new byte[BUFFER_SIZE];
            int bytesRead;
            AtomicLong totalBytesUploaded = new AtomicLong(0);
            long fileSize = file.length();

            while ((bytesRead = fileInputStream.read(buffer)) != -1) {
                dataOutputStream.write(buffer, 0, bytesRead);
                totalBytesUploaded.addAndGet(bytesRead);

                // 通知进度
                if (listener != null) {
                    listener.onProgress(totalBytesUploaded.get(), fileSize);
                }
            }

            dataOutputStream.writeBytes(LINE_END);
            dataOutputStream.writeBytes(TWO_HYPHENS + boundary + TWO_HYPHENS + LINE_END);
            dataOutputStream.flush();

            // 获取响应
            int responseCode = connection.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                String response = readResponse(connection);
                if (listener != null) {
                    listener.onSuccess(response);
                }
            } else {
                throw new IOException("上传失败，HTTP错误: " + responseCode);
            }

            Log.d(TAG, String.format("文件上传完成: %s, 响应码: %d", fileName, responseCode));

        } finally {
            // 关闭资源
            if (fileInputStream != null) {
                try {
                    fileInputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭FileInputStream失败", e);
                }
            }
            if (dataOutputStream != null) {
                try {
                    dataOutputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭DataOutputStream失败", e);
                }
            }
            if (outputStream != null) {
                try {
                    outputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭OutputStream失败", e);
                }
            }
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    // 上传多个文件
    private static void uploadMultipleFilesInternal(String url, File[] files, String[] fileParamNames,
                                                   UploadListener listener) throws IOException {
        String boundary = "----" + System.currentTimeMillis();
        HttpURLConnection connection = null;
        OutputStream outputStream = null;
        DataOutputStream dataOutputStream = null;

        try {
            URL uploadUrl = new URL(url);
            connection = (HttpURLConnection) uploadUrl.openConnection();

            // 配置连接
            connection.setRequestMethod("POST");
            connection.setConnectTimeout(30000);
            connection.setReadTimeout(30000);
            connection.setDoInput(true);
            connection.setDoOutput(true);
            connection.setUseCaches(false);

            // 设置请求头
            connection.setRequestProperty("Connection", "Keep-Alive");
            connection.setRequestProperty("User-Agent", "AndroidApp/1.0");
            connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

            // 获取输出流
            outputStream = connection.getOutputStream();
            dataOutputStream = new DataOutputStream(outputStream);

            // 计算总文件大小
            long totalSize = 0;
            for (File file : files) {
                totalSize += file.length();
            }

            AtomicLong totalBytesUploaded = new AtomicLong(0);

            // 逐个写入文件
            for (int i = 0; i < files.length; i++) {
                File file = files[i];
                String fileParamName = i < fileParamNames.length ? fileParamNames[i] : "file_" + i;

                // 写入文件数据头
                dataOutputStream.writeBytes(TWO_HYPHENS + boundary + LINE_END);
                dataOutputStream.writeBytes("Content-Disposition: form-data; name=\"" +
                    fileParamName + "\"; filename=\"" + file.getName() + "\"" + LINE_END);
                dataOutputStream.writeBytes("Content-Type: application/octet-stream" + LINE_END);
                dataOutputStream.writeBytes(LINE_END);

                // 写入文件内容
                try (FileInputStream fileInputStream = new FileInputStream(file)) {
                    byte[] buffer = new byte[BUFFER_SIZE];
                    int bytesRead;

                    while ((bytesRead = fileInputStream.read(buffer)) != -1) {
                        dataOutputStream.write(buffer, 0, bytesRead);
                        totalBytesUploaded.addAndGet(bytesRead);

                        // 通知进度
                        if (listener != null) {
                            listener.onProgress(totalBytesUploaded.get(), totalSize);
                        }
                    }
                }

                dataOutputStream.writeBytes(LINE_END);
            }

            // 结束边界
            dataOutputStream.writeBytes(TWO_HYPHENS + boundary + TWO_HYPHENS + LINE_END);
            dataOutputStream.flush();

            // 获取响应
            int responseCode = connection.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                String response = readResponse(connection);
                if (listener != null) {
                    listener.onSuccess(response);
                }
            } else {
                throw new IOException("上传失败，HTTP错误: " + responseCode);
            }

            Log.d(TAG, String.format("多文件上传完成，文件数: %d, 响应码: %d", files.length, responseCode));

        } finally {
            // 关闭资源
            if (dataOutputStream != null) {
                try {
                    dataOutputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭DataOutputStream失败", e);
                }
            }
            if (outputStream != null) {
                try {
                    outputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭OutputStream失败", e);
                }
            }
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    // 读取响应
    private static String readResponse(HttpURLConnection connection) throws IOException {
        InputStream inputStream = connection.getInputStream();
        StringBuilder response = new StringBuilder();

        try {
            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                response.append(new String(buffer, 0, bytesRead, "UTF-8"));
            }
        } finally {
            try {
                inputStream.close();
            } catch (IOException e) {
                Log.e(TAG, "关闭InputStream失败", e);
            }
        }

        return response.toString();
    }
}
```

## 🚨 网络错误处理

### 错误处理机制

```java
// NetworkErrorHandler.java
import android.util.Log;

import java.io.IOException;
import java.net.ConnectException;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;
import java.util.HashMap;
import java.util.Map;

public class NetworkErrorHandler {
    private static final String TAG = "NetworkErrorHandler";
    private static final Map<Integer, String> HTTP_ERROR_MESSAGES = new HashMap<>();

    static {
        // 初始化HTTP错误消息
        HTTP_ERROR_MESSAGES.put(400, "请求错误");
        HTTP_ERROR_MESSAGES.put(401, "未授权访问");
        HTTP_ERROR_MESSAGES.put(403, "禁止访问");
        HTTP_ERROR_MESSAGES.put(404, "资源未找到");
        HTTP_ERROR_MESSAGES.put(405, "请求方法不允许");
        HTTP_ERROR_MESSAGES.put(408, "请求超时");
        HTTP_ERROR_MESSAGES.put(429, "请求过于频繁");
        HTTP_ERROR_MESSAGES.put(500, "服务器内部错误");
        HTTP_ERROR_MESSAGES.put(502, "网关错误");
        HTTP_ERROR_MESSAGES.put(503, "服务不可用");
        HTTP_ERROR_MESSAGES.put(504, "网关超时");
    }

    // 网络错误类型
    public enum ErrorType {
        NETWORK_UNAVAILABLE,      // 网络不可用
        CONNECTION_TIMEOUT,       // 连接超时
        READ_TIMEOUT,            // 读取超时
        CONNECTION_REFUSED,      // 连接被拒绝
        UNKNOWN_HOST,            // 主机未知
        HTTP_ERROR,              // HTTP错误
        JSON_PARSE_ERROR,        // JSON解析错误
        SSL_ERROR,               // SSL错误
        IO_ERROR,                // IO错误
        UNKNOWN_ERROR            // 未知错误
    }

    // 网络错误类
    public static class NetworkError extends Exception {
        private final ErrorType errorType;
        private final int httpStatusCode;
        private final String errorMessage;
        private final boolean retryable;

        public NetworkError(ErrorType errorType, String message, int httpStatusCode, boolean retryable) {
            super(message);
            this.errorType = errorType;
            this.httpStatusCode = httpStatusCode;
            this.errorMessage = message;
            this.retryable = retryable;
        }

        public NetworkError(ErrorType errorType, String message, boolean retryable) {
            this(errorType, message, -1, retryable);
        }

        public NetworkError(ErrorType errorType, String message) {
            this(errorType, message, false);
        }

        // Getters
        public ErrorType getErrorType() { return errorType; }
        public int getHttpStatusCode() { return httpStatusCode; }
        public String getErrorMessage() { return errorMessage; }
        public boolean isRetryable() { return retryable; }

        @Override
        public String toString() {
            return String.format("NetworkError{type=%s, message='%s', httpCode=%d, retryable=%s}",
                errorType, errorMessage, httpStatusCode, retryable);
        }
    }

    // 分析错误
    public static NetworkError analyzeError(Exception e) {
        Log.e(TAG, "分析网络错误", e);

        if (e instanceof NetworkError) {
            return (NetworkError) e;
        }

        if (e instanceof SocketTimeoutException) {
            return new NetworkError(ErrorType.CONNECTION_TIMEOUT, "连接超时", true);
        }

        if (e instanceof ConnectException) {
            return new NetworkError(ErrorType.CONNECTION_REFUSED, "连接被拒绝", true);
        }

        if (e instanceof UnknownHostException) {
            return new NetworkError(ErrorType.UNKNOWN_HOST, "主机未知", false);
        }

        if (e instanceof IOException) {
            String message = e.getMessage();
            if (message != null) {
                if (message.contains("timeout") || message.contains("Timeout")) {
                    return new NetworkError(ErrorType.READ_TIMEOUT, "读取超时", true);
                }
                if (message.contains("SSL") || message.contains("SSLHandshakeException")) {
                    return new NetworkError(ErrorType.SSL_ERROR, "SSL连接错误", false);
                }
            }
            return new NetworkError(ErrorType.IO_ERROR, "IO错误: " + message, true);
        }

        return new NetworkError(ErrorType.UNKNOWN_ERROR, "未知错误: " + e.getMessage(), false);
    }

    // 分析HTTP错误
    public static NetworkError analyzeHttpError(int statusCode, String errorMessage) {
        String message = HTTP_ERROR_MESSAGES.getOrDefault(statusCode, "HTTP错误: " + statusCode);
        if (errorMessage != null) {
            message += " - " + errorMessage;
        }

        boolean retryable = isRetryableHttpError(statusCode);
        return new NetworkError(ErrorType.HTTP_ERROR, message, statusCode, retryable);
    }

    // 判断HTTP错误是否可重试
    private static boolean isRetryableHttpError(int statusCode) {
        return statusCode == 408 || // 请求超时
               statusCode == 429 || // 请求过于频繁
               statusCode == 500 || // 服务器内部错误
               statusCode == 502 || // 网关错误
               statusCode == 503 || // 服务不可用
               statusCode == 504;   // 网关超时
    }

    // 获取用户友好的错误消息
    public static String getUserFriendlyMessage(NetworkError error) {
        switch (error.getErrorType()) {
            case NETWORK_UNAVAILABLE:
                return "网络不可用，请检查网络连接";
            case CONNECTION_TIMEOUT:
            case READ_TIMEOUT:
                return "连接超时，请稍后重试";
            case CONNECTION_REFUSED:
                return "无法连接到服务器";
            case UNKNOWN_HOST:
                return "服务器地址无法解析";
            case HTTP_ERROR:
                if (error.getHttpStatusCode() == 401) {
                    return "登录已过期，请重新登录";
                } else if (error.getHttpStatusCode() == 403) {
                    return "没有访问权限";
                } else if (error.getHttpStatusCode() == 404) {
                    return "请求的资源不存在";
                } else if (error.getHttpStatusCode() == 429) {
                    return "请求过于频繁，请稍后重试";
                } else if (error.getHttpStatusCode() >= 500) {
                    return "服务器错误，请稍后重试";
                } else {
                    return error.getErrorMessage();
                }
            case JSON_PARSE_ERROR:
                return "数据解析失败";
            case SSL_ERROR:
                return "安全连接失败";
            case IO_ERROR:
                return "网络通信错误";
            case UNKNOWN_ERROR:
            default:
                return "未知错误，请稍后重试";
        }
    }

    // 错误重试策略
    public static class RetryStrategy {
        private final int maxRetries;
        private final long initialDelay;
        private final double backoffMultiplier;
        private final long maxDelay;

        public RetryStrategy(int maxRetries, long initialDelay, double backoffMultiplier, long maxDelay) {
            this.maxRetries = maxRetries;
            this.initialDelay = initialDelay;
            this.backoffMultiplier = backoffMultiplier;
            this.maxDelay = maxDelay;
        }

        // 预定义策略
        public static RetryStrategy getConservativeStrategy() {
            return new RetryStrategy(3, 1000, 2.0, 10000);
        }

        public static RetryStrategy getAggressiveStrategy() {
            return new RetryStrategy(5, 500, 1.5, 5000);
        }

        public static RetryStrategy getLinearStrategy() {
            return new RetryStrategy(3, 1000, 1.0, 3000);
        }

        // 计算重试延迟
        public long calculateRetryDelay(int retryCount) {
            long delay = (long) (initialDelay * Math.pow(backoffMultiplier, retryCount));
            return Math.min(delay, maxDelay);
        }

        // 判断是否应该重试
        public boolean shouldRetry(NetworkError error, int currentRetryCount) {
            return currentRetryCount < maxRetries && error.isRetryable();
        }

        // Getters
        public int getMaxRetries() { return maxRetries; }
        public long getInitialDelay() { return initialDelay; }
        public double getBackoffMultiplier() { return backoffMultiplier; }
        public long getMaxDelay() { return maxDelay; }
    }
}
```

## 📱 实践示例：网络请求管理器

### 综合网络请求管理器

```java
// NetworkManager.java
import android.content.Context;
import android.util.Log;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class NetworkManager {
    private static final String TAG = "NetworkManager";
    private static NetworkManager instance;

    private final Context context;
    private final OkHttpClient httpClient;
    private final HttpInterceptor httpInterceptor;
    private final TimeoutManager timeoutManager;
    private final ConnectionPoolManager connectionPoolManager;
    private final Map<String, Call> activeCalls;
    private final AtomicLong requestIdGenerator;

    private NetworkManager(Context context) {
        this.context = context.getApplicationContext();
        this.httpClient = createHttpClient();
        this.httpInterceptor = new HttpInterceptor();
        this.timeoutManager = new TimeoutManager();
        this.connectionPoolManager = ConnectionPoolManager.getInstance();
        this.activeCalls = new ConcurrentHashMap<>();
        this.requestIdGenerator = new AtomicLong(0);

        initInterceptors();
    }

    public static synchronized NetworkManager getInstance(Context context) {
        if (instance == null) {
            instance = new NetworkManager(context);
        }
        return instance;
    }

    // 创建OkHttpClient
    private OkHttpClient createHttpClient() {
        return new OkHttpClient.Builder()
            .connectTimeout(30000, java.util.concurrent.TimeUnit.MILLISECONDS)
            .readTimeout(60000, java.util.concurrent.TimeUnit.MILLISECONDS)
            .writeTimeout(30000, java.util.concurrent.TimeUnit.MILLISECONDS)
            .retryOnConnectionFailure(true)
            .build();
    }

    // 初始化拦截器
    private void initInterceptors() {
        // 添加User-Agent拦截器
        httpInterceptor.addRequestInterceptor(
            new HttpInterceptor.UserAgentInterceptor("AndroidApp/1.0")
        );

        // 添加日志拦截器
        httpInterceptor.addRequestInterceptor(new HttpInterceptor.LoggingInterceptor(true, false));
        httpInterceptor.addResponseInterceptor(new HttpInterceptor.LoggingInterceptor(true, false));

        // 添加缓存拦截器
        httpInterceptor.addResponseInterceptor(new HttpInterceptor.CacheInterceptor());
    }

    // 设置认证令牌
    public void setAuthToken(String token) {
        // 移除旧的认证拦截器
        httpInterceptor.getRequestInterceptors().removeIf(interceptor ->
            interceptor instanceof HttpInterceptor.AuthInterceptor);

        // 添加新的认证拦截器
        if (token != null && !token.isEmpty()) {
            httpInterceptor.addRequestInterceptor(new HttpInterceptor.AuthInterceptor(token));
        }
    }

    // GET请求
    public String get(String url) throws NetworkErrorHandler.NetworkError {
        return get(url, new HashMap<>());
    }

    public String get(String url, Map<String, String> headers) throws NetworkErrorHandler.NetworkError {
        String requestId = generateRequestId();
        try {
            Log.d(TAG, String.format("执行GET请求: %s [ID: %s]", url, requestId));

            // 创建请求数据
            HttpInterceptor.RequestData requestData = new HttpInterceptor.RequestData(
                url, "GET", headers, null);

            // 执行请求拦截
            requestData = httpInterceptor.interceptRequest(requestData);

            // 执行请求
            return executeRequest(requestId, requestData);

        } catch (Exception e) {
            Log.e(TAG, String.format("GET请求失败: %s [ID: %s]", url, requestId), e);
            throw NetworkErrorHandler.analyzeError(e);
        }
    }

    // 异步GET请求
    public void getAsync(String url, NetworkCallback callback) {
        getAsync(url, new HashMap<>(), callback);
    }

    public void getAsync(String url, Map<String, String> headers, NetworkCallback callback) {
        String requestId = generateRequestId();
        Log.d(TAG, String.format("执行异步GET请求: %s [ID: %s]", url, requestId));

        try {
            // 创建请求数据
            HttpInterceptor.RequestData requestData = new HttpInterceptor.RequestData(
                url, "GET", headers, null);

            // 执行请求拦截
            requestData = httpInterceptor.interceptRequest(requestData);

            // 创建OkHttp请求
            Request.Builder requestBuilder = new Request.Builder()
                .url(requestData.getUrl())
                .get();

            // 添加请求头
            for (Map.Entry<String, String> entry : requestData.getHeaders().entrySet()) {
                requestBuilder.addHeader(entry.getKey(), entry.getValue());
            }

            Request request = requestBuilder.build();
            Call call = httpClient.newCall(request);

            // 添加到活跃请求列表
            activeCalls.put(requestId, call);

            // 执行异步请求
            call.enqueue(new Callback() {
                @Override
                public void onFailure(Call call, IOException e) {
                    activeCalls.remove(requestId);
                    NetworkErrorHandler.NetworkError error = NetworkErrorHandler.analyzeError(e);
                    Log.e(TAG, String.format("异步GET请求失败: %s [ID: %s]", url, requestId), e);

                    if (callback != null) {
                        callback.onError(error);
                    }
                }

                @Override
                public void onResponse(Call call, Response response) throws IOException {
                    activeCalls.remove(requestId);

                    try {
                        if (!response.isSuccessful()) {
                            NetworkErrorHandler.NetworkError error = NetworkErrorHandler.analyzeHttpError(
                                response.code(), response.message());
                            Log.e(TAG, String.format("HTTP错误: %s [ID: %s]", error.toString(), requestId));

                            if (callback != null) {
                                callback.onError(error);
                            }
                            return;
                        }

                        String responseBody = response.body().string();

                        // 创建响应数据
                        Map<String, String> responseHeaders = new HashMap<>();
                        for (String name : response.headers().names()) {
                            responseHeaders.put(name, response.header(name));
                        }

                        HttpInterceptor.ResponseData responseData = new HttpInterceptor.ResponseData(
                            response.code(), responseHeaders, responseBody.getBytes());

                        // 执行响应拦截
                        responseData = httpInterceptor.interceptResponse(responseData);

                        if (callback != null) {
                            callback.onSuccess(new String(responseData.getBody()));
                        }

                        Log.d(TAG, String.format("异步GET请求成功: %s [ID: %s], 响应大小: %d",
                            url, requestId, responseBody.length()));

                    } catch (Exception e) {
                        NetworkErrorHandler.NetworkError error = NetworkErrorHandler.analyzeError(e);
                        Log.e(TAG, String.format("处理响应失败: %s [ID: %s]", url, requestId), e);

                        if (callback != null) {
                            callback.onError(error);
                        }
                    } finally {
                        response.close();
                    }
                }
            });

        } catch (Exception e) {
            activeCalls.remove(requestId);
            NetworkErrorHandler.NetworkError error = NetworkErrorHandler.analyzeError(e);
            Log.e(TAG, String.format("创建异步GET请求失败: %s [ID: %s]", url, requestId), e);

            if (callback != null) {
                callback.onError(error);
            }
        }
    }

    // POST请求
    public String post(String url, String data) throws NetworkErrorHandler.NetworkError {
        return post(url, data, "application/json", new HashMap<>());
    }

    public String post(String url, String data, String contentType, Map<String, String> headers)
        throws NetworkErrorHandler.NetworkError {

        String requestId = generateRequestId();
        try {
            Log.d(TAG, String.format("执行POST请求: %s [ID: %s]", url, requestId));

            // 创建请求数据
            HttpInterceptor.RequestData requestData = new HttpInterceptor.RequestData(
                url, "POST", headers, data != null ? data.getBytes() : null);

            // 设置Content-Type
            if (contentType != null) {
                requestData.addHeader("Content-Type", contentType);
            }

            // 执行请求拦截
            requestData = httpInterceptor.interceptRequest(requestData);

            // 执行请求
            return executeRequest(requestId, requestData);

        } catch (Exception e) {
            Log.e(TAG, String.format("POST请求失败: %s [ID: %s]", url, requestId), e);
            throw NetworkErrorHandler.analyzeError(e);
        }
    }

    // 执行请求
    private String executeRequest(String requestId, HttpInterceptor.RequestData requestData)
        throws IOException {

        // 创建OkHttp请求
        Request.Builder requestBuilder = new Request.Builder()
            .url(requestData.getUrl());

        // 设置请求方法
        switch (requestData.getMethod()) {
            case "GET":
                requestBuilder.get();
                break;
            case "POST":
                MediaType mediaType = MediaType.parse(requestData.getHeader("Content-Type"));
                RequestBody requestBody = RequestBody.create(mediaType, requestData.getBody());
                requestBuilder.post(requestBody);
                break;
            // 可以添加其他HTTP方法
        }

        // 添加请求头
        for (Map.Entry<String, String> entry : requestData.getHeaders().entrySet()) {
            requestBuilder.addHeader(entry.getKey(), entry.getValue());
        }

        Request request = requestBuilder.build();
        Call call = httpClient.newCall(request);

        try {
            // 执行请求
            Response response = call.execute();

            if (!response.isSuccessful()) {
                throw NetworkErrorHandler.analyzeHttpError(response.code(), response.message());
            }

            // 创建响应数据
            Map<String, String> responseHeaders = new HashMap<>();
            for (String name : response.headers().names()) {
                responseHeaders.put(name, response.header(name));
            }

            HttpInterceptor.ResponseData responseData = new HttpInterceptor.ResponseData(
                response.code(), responseHeaders, response.body().bytes());

            // 执行响应拦截
            responseData = httpInterceptor.interceptResponse(responseData);

            String responseBody = new String(responseData.getBody());
            Log.d(TAG, String.format("请求成功: %s [ID: %s], 响应大小: %d",
                requestData.getUrl(), requestId, responseBody.length()));

            return responseBody;

        } finally {
            // response会自动关闭
        }
    }

    // 取消请求
    public void cancelRequest(String requestId) {
        Call call = activeCalls.remove(requestId);
        if (call != null) {
            call.cancel();
            Log.d(TAG, String.format("请求已取消: [ID: %s]", requestId));
        }
    }

    // 取消所有请求
    public void cancelAllRequests() {
        for (Map.Entry<String, Call> entry : activeCalls.entrySet()) {
            entry.getValue().cancel();
            Log.d(TAG, String.format("请求已取消: [ID: %s]", entry.getKey()));
        }
        activeCalls.clear();
    }

    // 生成请求ID
    private String generateRequestId() {
        return "REQ_" + requestIdGenerator.incrementAndGet();
    }

    // 网络回调接口
    public interface NetworkCallback {
        void onSuccess(String response);
        void onError(NetworkErrorHandler.NetworkError error);
    }

    // 获取活跃请求数量
    public int getActiveRequestCount() {
        return activeCalls.size();
    }

    // 获取连接池状态
    public ConnectionPoolManager.PoolStatus getConnectionPoolStatus() {
        return connectionPoolManager.getPoolStatus();
    }

    // 关闭网络管理器
    public void shutdown() {
        cancelAllRequests();
        timeoutManager.shutdown();
        connectionPoolManager.shutdown();
        Log.d(TAG, "网络管理器已关闭");
    }
}
```

## 📊 网络请求监控

### 网络性能监控器

```java
// NetworkMonitor.java
import android.util.Log;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public class NetworkMonitor {
    private static final String TAG = "NetworkMonitor";
    private static NetworkMonitor instance;

    private final Map<String, RequestMetrics> requestMetricsMap;
    private final AtomicLong totalRequests;
    private final AtomicLong successfulRequests;
    private final AtomicLong failedRequests;
    private final AtomicLong totalResponseTime;
    private final AtomicLong totalBytesReceived;
    private final AtomicLong totalBytesSent;

    private NetworkMonitor() {
        this.requestMetricsMap = new ConcurrentHashMap<>();
        this.totalRequests = new AtomicLong(0);
        this.successfulRequests = new AtomicLong(0);
        this.failedRequests = new AtomicLong(0);
        this.totalResponseTime = new AtomicLong(0);
        this.totalBytesReceived = new AtomicLong(0);
        this.totalBytesSent = new AtomicLong(0);
    }

    public static synchronized NetworkMonitor getInstance() {
        if (instance == null) {
            instance = new NetworkMonitor();
        }
        return instance;
    }

    // 开始监控请求
    public void startRequest(String requestId, String url, String method) {
        RequestMetrics metrics = new RequestMetrics(requestId, url, method);
        metrics.setStartTime(System.currentTimeMillis());
        requestMetricsMap.put(requestId, metrics);
        totalRequests.incrementAndGet();

        Log.d(TAG, String.format("开始监控请求: %s %s [ID: %s]", method, url, requestId));
    }

    // 完成请求
    public void completeRequest(String requestId, int statusCode, long responseSize) {
        RequestMetrics metrics = requestMetricsMap.get(requestId);
        if (metrics != null) {
            metrics.setEndTime(System.currentTimeMillis());
            metrics.setStatusCode(statusCode);
            metrics.setResponseSize(responseSize);
            metrics.setSuccess(statusCode >= 200 && statusCode < 300);

            long responseTime = metrics.getEndTime() - metrics.getStartTime();
            totalResponseTime.addAndGet(responseTime);
            totalBytesReceived.addAndGet(responseSize);

            if (metrics.isSuccess()) {
                successfulRequests.incrementAndGet();
            } else {
                failedRequests.incrementAndGet();
            }

            Log.d(TAG, String.format("请求完成: %s [ID: %s], 状态码: %d, 响应时间: %dms, 响应大小: %d bytes",
                metrics.getUrl(), requestId, statusCode, responseTime, responseSize));
        }
    }

    // 记录发送字节数
    public void recordBytesSent(String requestId, long bytesSent) {
        RequestMetrics metrics = requestMetricsMap.get(requestId);
        if (metrics != null) {
            metrics.setRequestSize(bytesSent);
            totalBytesSent.addAndGet(bytesSent);
        }
    }

    // 记录请求错误
    public void recordRequestError(String requestId, String error) {
        RequestMetrics metrics = requestMetricsMap.get(requestId);
        if (metrics != null) {
            metrics.setEndTime(System.currentTimeMillis());
            metrics.setError(error);
            metrics.setSuccess(false);

            failedRequests.incrementAndGet();

            Log.e(TAG, String.format("请求错误: %s [ID: %s], 错误: %s",
                metrics != null ? metrics.getUrl() : "unknown", requestId, error));
        }
    }

    // 获取网络统计
    public NetworkStatistics getNetworkStatistics() {
        NetworkStatistics stats = new NetworkStatistics();

        stats.setTotalRequests(totalRequests.get());
        stats.setSuccessfulRequests(successfulRequests.get());
        stats.setFailedRequests(failedRequests.get());
        stats.setTotalResponseTime(totalResponseTime.get());
        stats.setTotalBytesReceived(totalBytesReceived.get());
        stats.setTotalBytesSent(totalBytesSent.get());

        if (stats.getTotalRequests() > 0) {
            stats.setSuccessRate((stats.getSuccessfulRequests() * 100.0) / stats.getTotalRequests());
            stats.setAverageResponseTime(stats.getTotalResponseTime() / stats.getTotalRequests());
        }

        // 按URL统计
        Map<String, UrlStatistics> urlStatsMap = new HashMap<>();
        for (RequestMetrics metrics : requestMetricsMap.values()) {
            String url = metrics.getUrl();
            UrlStatistics urlStats = urlStatsMap.computeIfAbsent(url, UrlStatistics::new);
            urlStats.addMetrics(metrics);
        }
        stats.setUrlStatisticsMap(urlStatsMap);

        return stats;
    }

    // 清理旧的监控数据
    public void cleanupOldMetrics(long cutoffTime) {
        requestMetricsMap.entrySet().removeIf(entry ->
            entry.getValue().getStartTime() < cutoffTime);
        Log.d(TAG, "清理旧的监控数据完成");
    }

    // 重置统计
    public void resetStatistics() {
        requestMetricsMap.clear();
        totalRequests.set(0);
        successfulRequests.set(0);
        failedRequests.set(0);
        totalResponseTime.set(0);
        totalBytesReceived.set(0);
        totalBytesSent.set(0);
        Log.d(TAG, "网络统计数据已重置");
    }

    // 请求指标
    public static class RequestMetrics {
        private final String requestId;
        private final String url;
        private final String method;
        private long startTime;
        private long endTime;
        private int statusCode;
        private long requestSize;
        private long responseSize;
        private boolean success;
        private String error;

        public RequestMetrics(String requestId, String url, String method) {
            this.requestId = requestId;
            this.url = url;
            this.method = method;
        }

        // Getters and Setters
        public long getResponseTime() { return endTime - startTime; }

        public String getRequestId() { return requestId; }
        public String getUrl() { return url; }
        public String getMethod() { return method; }
        public long getStartTime() { return startTime; }
        public void setStartTime(long startTime) { this.startTime = startTime; }
        public long getEndTime() { return endTime; }
        public void setEndTime(long endTime) { this.endTime = endTime; }
        public int getStatusCode() { return statusCode; }
        public void setStatusCode(int statusCode) { this.statusCode = statusCode; }
        public long getRequestSize() { return requestSize; }
        public void setRequestSize(long requestSize) { this.requestSize = requestSize; }
        public long getResponseSize() { return responseSize; }
        public void setResponseSize(long responseSize) { this.responseSize = responseSize; }
        public boolean isSuccess() { return success; }
        public void setSuccess(boolean success) { this.success = success; }
        public String getError() { return error; }
        public void setError(String error) { this.error = error; }
    }

    // URL统计
    public static class UrlStatistics {
        private final String url;
        private long totalRequests;
        private long successfulRequests;
        private long totalResponseTime;
        private long totalBytesReceived;
        private long totalBytesSent;

        public UrlStatistics(String url) {
            this.url = url;
        }

        public void addMetrics(RequestMetrics metrics) {
            totalRequests++;
            if (metrics.isSuccess()) {
                successfulRequests++;
                totalResponseTime += metrics.getResponseTime();
                totalBytesReceived += metrics.getResponseSize();
                totalBytesSent += metrics.getRequestSize();
            }
        }

        // Getters
        public String getUrl() { return url; }
        public long getTotalRequests() { return totalRequests; }
        public long getSuccessfulRequests() { return successfulRequests; }
        public double getSuccessRate() {
            return totalRequests > 0 ? (successfulRequests * 100.0 / totalRequests) : 0;
        }
        public long getAverageResponseTime() {
            return successfulRequests > 0 ? (totalResponseTime / successfulRequests) : 0;
        }
        public long getTotalBytesReceived() { return totalBytesReceived; }
        public long getTotalBytesSent() { return totalBytesSent; }
    }

    // 网络统计
    public static class NetworkStatistics {
        private long totalRequests;
        private long successfulRequests;
        private long failedRequests;
        private long totalResponseTime;
        private long totalBytesReceived;
        private long totalBytesSent;
        private double successRate;
        private long averageResponseTime;
        private Map<String, UrlStatistics> urlStatisticsMap;

        // Getters and Setters
        public long getTotalRequests() { return totalRequests; }
        public void setTotalRequests(long totalRequests) { this.totalRequests = totalRequests; }
        public long getSuccessfulRequests() { return successfulRequests; }
        public void setSuccessfulRequests(long successfulRequests) { this.successfulRequests = successfulRequests; }
        public long getFailedRequests() { return failedRequests; }
        public void setFailedRequests(long failedRequests) { this.failedRequests = failedRequests; }
        public long getTotalResponseTime() { return totalResponseTime; }
        public void setTotalResponseTime(long totalResponseTime) { this.totalResponseTime = totalResponseTime; }
        public long getTotalBytesReceived() { return totalBytesReceived; }
        public void setTotalBytesReceived(long totalBytesReceived) { this.totalBytesReceived = totalBytesReceived; }
        public long getTotalBytesSent() { return totalBytesSent; }
        public void setTotalBytesSent(long totalBytesSent) { this.totalBytesSent = totalBytesSent; }
        public double getSuccessRate() { return successRate; }
        public void setSuccessRate(double successRate) { this.successRate = successRate; }
        public long getAverageResponseTime() { return averageResponseTime; }
        public void setAverageResponseTime(long averageResponseTime) { this.averageResponseTime = averageResponseTime; }
        public Map<String, UrlStatistics> getUrlStatisticsMap() { return urlStatisticsMap; }
        public void setUrlStatisticsMap(Map<String, UrlStatistics> urlStatisticsMap) { this.urlStatisticsMap = urlStatisticsMap; }

        @Override
        public String toString() {
            return String.format("NetworkStatistics{total=%d, success=%d, failed=%d, successRate=%.2f%%, avgResponseTime=%dms, bytesReceived=%d, bytesSent=%d}",
                totalRequests, successfulRequests, failedRequests, successRate, averageResponseTime, totalBytesReceived, totalBytesSent);
        }
    }
}
```

## 📝 本章小结

### 核心知识点

1. **HttpURLConnection使用**
   - 基本GET/POST请求实现
   - 文件上传下载功能
   - 连接配置和资源管理

2. **连接池和超时处理**
   - 连接池管理优化性能
   - 超时配置和重试策略
   - 网络请求并发控制

3. **请求和响应拦截**
   - 拦截器模式实现
   - 请求头和响应头处理
   - 缓存和日志拦截器

4. **文件上传下载**
   - 单文件和多文件上传
   - 断点续传下载
   - 进度监听和状态管理

5. **网络错误处理**
   - 错误类型分析和分类
   - 重试策略和错误恢复
   - 用户友好的错误提示

6. **网络监控**
   - 请求性能监控
   - 统计数据收集
   - 网络状态分析

### 实践建议

1. **HttpURLConnection使用**
   - 优先使用OkHttp等现代网络库
   - 正确管理连接和资源
   - 实现超时和重试机制

2. **性能优化**
   - 使用连接池提高性能
   - 配置合适的超时时间
   - 实现智能重试策略

3. **错误处理**
   - 详细分析网络错误类型
   - 提供用户友好的错误提示
   - 实现自动重试机制

4. **监控和调试**
   - 记录详细的网络请求日志
   - 监控网络性能指标
   - 定期分析统计数据

### 常见问题解决

1. **网络连接失败**
   - 检查网络权限配置
   - 验证URL格式和可达性
   - 配置合适的超时时间

2. **文件上传下载问题**
   - 检查文件路径和权限
   - 处理大文件分块传输
   - 实现进度监听功能

3. **性能问题**
   - 优化连接池配置
   - 减少不必要的网络请求
   - 实现请求缓存机制

4. **错误处理问题**
   - 正确分析错误类型
   - 实现合适的重试策略
   - 提供清晰的错误提示

通过本章的学习，你掌握了Android中HTTP请求和响应处理的完整知识体系，包括HttpURLConnection的使用、连接池管理、超时处理、拦截器实现、文件上传下载、错误处理和网络监控。这些技能为构建高质量的网络应用提供了坚实的基础。