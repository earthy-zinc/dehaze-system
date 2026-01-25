# dehaze-java JMeter接口测试方案

## 📖 目录

- [概述](#概述)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [测试用例说明](#测试用例说明)
- [使用指南](#使用指南)
- [测试报告](#测试报告)
- [常见问题](#常见问题)

---

## 概述

本测试方案为dehaze-java项目提供了完整的JMeter接口测试解决方案，包括：

- ✅ 自动化安装脚本
- ✅ 完整的接口测试计划
- ✅ 性能压力测试脚本
- ✅ 测试数据管理
- ✅ 自动化报告生成

---

## 环境要求

### 必需环境

- **操作系统**: Linux (推荐 TencentOS / CentOS / Ubuntu)
- **Java**: JDK 8 或更高版本
- **权限**: Root 或 sudo 权限（用于安装）

### 检查Java环境

```bash
java -version
```

如果未安装Java，请先安装：

```bash
# TencentOS/CentOS
sudo yum install -y java-17-openjdk java-17-openjdk-devel

# Ubuntu
sudo apt-get install -y openjdk-17-jdk
```

---

## 安装步骤

### 1. 下载安装脚本

```bash
cd /data/workspace/dehaze-system/jmeter
```

### 2. 执行安装脚本

```bash
chmod +x install-jmeter.sh
sudo ./install-jmeter.sh
```

JMeter将安装在 `/usr/local/jmeter` 目录中。

### 3. 配置环境变量

```bash
source /etc/profile.d/jmeter.sh
```

### 4. 验证安装

```bash
jmeter --version
```

成功输出示例：

```
Version 5.6.3
Copyright...
```

---

## 测试用例说明

### 测试覆盖的接口

#### 1. 认证接口 (`AuthController`)

| 接口                     | 方法     | 测试用例  | 断言                                  |
|------------------------|--------|-------|-------------------------------------|
| `/api/v1/auth/captcha` | GET    | 获取验证码 | 状态码200、响应包含captchaKey和captchaImage  |
| `/api/v1/auth/login`   | POST   | 用户登录  | 状态码200、响应包含accessToken和refreshToken |
| `/api/v1/auth/logout`  | DELETE | 用户注销  | 状态码200                              |

#### 2. 文件管理接口 (`FileController`)

| 接口                                | 方法     | 测试用例   | 断言                               |
|-----------------------------------|--------|--------|----------------------------------|
| `/api/v1/files/page`              | GET    | 分页查询文件 | 状态码200、响应包含records、total、pageNum |
| `/api/v1/files`                   | POST   | 文件上传   | 状态码200、响应包含id、name、url           |
| `/api/v1/files/{fileId}`          | GET    | 获取文件详情 | 状态码200                           |
| `/api/v1/files/check`             | GET    | 文件校验   | 状态码200                           |
| `/api/v1/files`                   | DELETE | 删除文件   | 状态码200                           |
| `/api/v1/files/download/{taskId}` | GET    | 下载文件   | 状态码200                           |

### 测试用例组织

```
jmeter/
├── test-plans/
│   └── dehaze-api-test.jmx          # 主测试计划
├── configs/
│   └── test-env.properties           # 测试环境配置
├── data/
│   └── test-users.csv               # 测试用户数据
├── reports/                         # 测试报告输出目录（自动创建）
└── scripts/
    ├── install-jmeter.sh            # 安装脚本
    ├── run-test.sh                  # 测试执行脚本
    └── run-performance-test.sh      # 性能测试脚本
```

---

## 使用指南

### 配置测试环境

编辑配置文件 `configs/test-env.properties`:

```properties
# 服务器配置
TEST_BASE_URL=http://localhost:8080

# 测试账号
TEST_USERNAME=admin
TEST_PASSWORD=123456

# 超时配置
CONNECT_TIMEOUT=10000
READ_TIMEOUT=30000
```

### 准备测试数据

创建测试文件用于上传测试：

```bash
echo "This is a test file for JMeter upload test." > /tmp/test-file.txt
```

### 执行接口测试

#### 方式1: 使用执行脚本（推荐）

```bash
cd /data/workspace/dehaze-system/jmeter

# 赋予执行权限
chmod +x run-test.sh

# 执行测试
./run-test.sh
```

#### 方式2: 使用JMeter命令

```bash
# GUI模式（用于调试）
jmeter -t test-plans/dehaze-api-test.jmx

# 命令行模式（推荐）
jmeter -n -t test-plans/dehaze-api-test.jmx \
       -l reports/result-$(date +%Y%m%d-%H%M%S).jtl \
       -e -o reports/html-report-$(date +%Y%m%d-%H%M%S)
```

### 执行性能压力测试

```bash
cd /data/workspace/dehaze-system/jmeter

# 赋予执行权限
chmod +x run-performance-test.sh

# 查看帮助
./run-performance-test.sh --help

# 执行性能测试（默认参数）
./run-performance-test.sh

# 自定义参数执行
# 参数1: 并发用户数
# 参数2: 循环次数
# 参数3: 启动时间（秒）
./run-performance-test.sh 50 200 30
```

性能测试参数说明：

- **并发用户数**: 模拟同时访问的用户数量
- **循环次数**: 每个用户发送的请求次数
- **启动时间**: 所有用户启动完成的时间（秒）
- **总请求数**: 并发用户数 × 循环次数

---

## 测试报告

### 查看测试报告

测试执行完成后，HTML报告会自动生成在 `reports/` 目录下：

```bash
# 查看最新的报告
ls -lt reports/ | head -5

# 使用浏览器打开报告
xdg-open reports/html-report-xxxxxx/index.html
```

### 报告说明

JMeter生成的HTML报告包含以下内容：

1. **Dashboard**: 测试概览
    - 平均响应时间
    - 吞吐量
    - 错误率
    - 活跃线程数

2. **Charts**: 性能图表
    - 响应时间趋势图
    - 活跃线程趋势图
    - 吞吐量趋势图

3. **Statistics**: 统计数据
    - 请求总数
    - 成功/失败数量
    - 最小/平均/最大响应时间
    - 百分位响应时间

4. **Errors**: 错误详情
    - 失败的请求列表
    - 错误原因分析

---

## 常见问题

### 1. JMeter安装失败

**问题**: 提示"未检测到Java环境"

**解决方案**:

```bash
# 安装Java
sudo yum install -y java-17-openjdk java-17-openjdk-devel

# 验证Java安装
java -version
```

### 2. 测试连接失败

**问题**: 所有请求都显示连接失败

**解决方案**:

1. 检查dehaze-java服务是否启动
2. 检查 `configs/test-env.properties` 中的 `TEST_BASE_URL` 是否正确
3. 检查防火墙设置
4. 检查服务器端口是否开放

### 3. 认证接口失败

**问题**: 登录接口返回401或403

**解决方案**:

1. 确认测试用户名和密码正确
2. 检查Spring Security配置
3. 查看后端日志获取详细错误信息

### 4. 文件上传失败

**问题**: 文件上传接口测试失败

**解决方案**:

```bash
# 确保测试文件存在
ls -la /tmp/test-file.txt

# 如果不存在，创建测试文件
echo "test content" > /tmp/test-file.txt
```

### 5. 内存不足

**问题**: 测试执行过程中提示"Out of Memory"

**解决方案**:
修改JMeter堆内存配置：

```bash
vim $JMETER_HOME/bin/jmeter

# 修改HEAP参数（增加到4GB）
HEAP="-Xms2g -Xmx4g -XMaxMetaspaceSize=512m"
```

### 6. 性能测试结果不理想

**问题**: 性能测试显示成功率低或响应时间长

**解决方案**:

1. 降低并发用户数，逐步增加
2. 检查后端服务性能
3. 检查数据库连接池配置
4. 检查服务器资源使用情况（CPU、内存、磁盘IO）

---

## 高级配置

### 修改测试计划

使用JMeter GUI编辑测试计划：

```bash
cd /data/workspace/dehaze-system/jmeter
jmeter -t test-plans/dehaze-api-test.jmx
```

在GUI中可以：

- 添加新的测试用例
- 修改断言规则
- 调整测试参数
- 添加定时器、监听器等

### 配置代理录制

如果需要录制实际的请求：

1. 在JMeter中添加HTTP(S) Test Script Recorder
2. 配置浏览器使用JMeter作为代理
3. 访问应用，JMeter自动录制请求

### 集成到CI/CD

可以将测试集成到CI/CD流程中：

```bash
# 在Jenkins/GitLab CI中执行
cd /data/workspace/dehaze-system/jmeter

# 执行测试
./run-test.sh

# 检查测试结果
if [ $? -eq 0 ]; then
    echo "测试通过"
else
    echo "测试失败"
    exit 1
fi
```

---

## 参考资料

- [JMeter官方文档](https://jmeter.apache.org/usermanual/index.html)
- [JMeter性能测试最佳实践](https://jmeter.apache.org/usermanual/best-practices.html)
- [dehaze-java项目文档](../../README.md)

---

## 联系方式

如有问题，请联系项目维护团队。
