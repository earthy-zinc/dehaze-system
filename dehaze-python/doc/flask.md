Flask 应用在创建时默认包含一组基础配置项，这些配置项控制应用的行为，包括调试模式、安全设置、日志记录等。可以通过 `app.config`
属性访问和修改这些配置。以下是 Flask 创建时的默认配置项及其说明：

---

### **默认配置项列表**

| 配置项                             | 默认值                        | 说明                                                                    |
|---------------------------------|----------------------------|-----------------------------------------------------------------------|
| `DEBUG`                         | `False`                    | 是否启用调试模式。可以通过 `app.run(debug=True)` 显式启用调试。                           |
| `TESTING`                       | `False`                    | 是否启用测试模式。启用后，错误不会被报告到日志系统，并且可以在测试中进行特定的功能调用。                          |
| `SECRET_KEY`                    | `None`                     | 用于加密 session 和 cookies 的密钥。生产环境中必须设置此值以保证安全性。                         |
| `SESSION_COOKIE_NAME`           | `'session'`                | 浏览器中存储的 session cookie 的名称。                                           |
| `SESSION_COOKIE_DOMAIN`         | `None`                     | 设置 session cookie 的域名。如果未设置，则会根据请求自动推断。                               |
| `SESSION_COOKIE_PATH`           | `None`                     | 设置 session cookie 的路径。默认值为应用根路径。                                      |
| `SESSION_COOKIE_HTTPONLY`       | `True`                     | 是否将 session cookie 设置为 HTTP-only（无法通过 JavaScript 访问）。                 |
| `SESSION_COOKIE_SECURE`         | `False`                    | 是否将 session cookie 标记为仅在 HTTPS 下传输。                                   |
| `SESSION_COOKIE_SAMESITE`       | `'Lax'`                    | 控制 session cookie 的跨站请求行为。可设置为 `'Strict'`、`'Lax'` 或 `None`。           |
| `PERMANENT_SESSION_LIFETIME`    | `timedelta(days=31)`       | 设置 session 的默认过期时间。                                                   |
| `USE_X_SENDFILE`                | `False`                    | 是否启用 X-Sendfile 头支持，用于更高效地发送静态文件（依赖于服务器支持）。                           |
| `SERVER_NAME`                   | `None`                     | 应用的服务器名称和端口号（如 `example.com:5000`）。影响 URL 生成。                         |
| `APPLICATION_ROOT`              | `'/'`                      | 应用的 URL 前缀路径。默认根路径为 `'/'`。                                            |
| `PREFERRED_URL_SCHEME`          | `'http'`                   | 用于 URL 生成的默认协议（`http` 或 `https`）。                                     |
| `MAX_CONTENT_LENGTH`            | `None`                     | 限制 HTTP 请求的最大大小（以字节为单位）。超过限制的请求将返回 `413 Request Entity Too Large` 错误。 |
| `TRAP_HTTP_EXCEPTIONS`          | `False`                    | 是否将 HTTP 异常视为普通异常抛出，便于调试和处理。                                          |
| `TRAP_BAD_REQUEST_ERRORS`       | `None`                     | 是否在调试时捕获请求错误。默认与 `DEBUG` 设置一致。                                        |
| `JSON_AS_ASCII`                 | `True`                     | JSON 响应中的非 ASCII 字符是否被转义为 ASCII。                                      |
| `JSON_SORT_KEYS`                | `True`                     | 是否按键排序生成 JSON 响应。                                                     |
| `JSONIFY_PRETTYPRINT_REGULAR`   | `True`                     | 是否美化格式化 JSON 响应。在调试模式下默认启用。                                           |
| `JSONIFY_MIMETYPE`              | `'application/json'`       | JSON 响应的默认 MIME 类型。                                                   |
| `TEMPLATES_AUTO_RELOAD`         | `None`                     | 是否在开发模式下自动重新加载模板文件。默认根据 `DEBUG` 设置自动启用或禁用。                            |
| `EXPLAIN_TEMPLATE_LOADING`      | `False`                    | 是否显示模板加载的调试信息。                                                        |
| `PROPAGATE_EXCEPTIONS`          | `None`                     | 是否在调试模式下传播异常。默认值由 Flask 根据环境设置决定。                                     |
| `PRESERVE_CONTEXT_ON_EXCEPTION` | `None`                     | 是否在异常发生时保留上下文堆栈。通常用于调试模式。                                             |
| `LOGGER_NAME`                   | `'flask.app'`              | 默认日志记录器的名称。                                                           |
| `LOGGER_HANDLER_POLICY`         | `'always'`                 | 控制默认日志处理程序的行为（如 `'always'`、`'debug'`、`'never'`）。                      |
| `SEND_FILE_MAX_AGE_DEFAULT`     | `timedelta(seconds=43200)` | 默认的静态文件缓存时间。                                                          |
| `ENV`                           | `'production'`             | Flask 应用的运行环境（`'development'` 或 `'production'`）。                      |
| `DEBUG`                         | `False`                    | 是否启用调试模式。                                                             |
| `LOAD_ENV`                      | `True`                     | 是否加载 `.env` 文件中的配置。                                                   |
| `FLASK_ENV`                     | `'production'`             | 设置 Flask 应用的环境变量。可以为 `development` 或 `production`。                    |
| `FLASK_DEBUG`                   | `0`                        | 是否启用调试模式（通过环境变量）。                                                     |

---

### **访问和修改默认配置**

Flask 配置可以通过以下方式访问和修改：

#### **访问配置**

```python
from flask import Flask

app = Flask(__name__)
print(app.config['DEBUG'])  # 输出默认值 False
```

#### **修改配置**

1. **直接修改：**
   ```python
   app.config['DEBUG'] = True
   ```

2. **从配置对象加载：**
   创建配置类文件 `config.py`：
   ```python
   class Config:
       DEBUG = True
       SECRET_KEY = 'your_secret_key'
   ```

   然后加载：
   ```python
   app.config.from_object('config.Config')
   ```

3. **从环境变量加载：**
   ```python
   app.config.from_envvar('FLASK_SETTINGS')
   ```

4. **从文件加载：**
   ```python
   app.config.from_pyfile('config.py')
   ```

---

### **总结**

Flask 提供了一组合理的默认配置，覆盖了应用运行时的核心功能，比如调试模式、session 管理、JSON
处理和日志记录等。开发者可以根据需求对这些配置进行调整，以适配开发或生产环境。
