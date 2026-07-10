# JMeter快速使用指南

## 🚀 快速开始（5分钟上手）

### 前置条件

- ✅ Java 8+ 已安装
- ✅ dehaze-java 服务已启动（端口8080）
- ✅ Root 或 sudo 权限

### 一键启动（推荐）

```bash
cd /data/workspace/dehaze-system/jmeter

# 运行快速启动脚本（交互式菜单）
./quick-start.sh
```

**注意**: JMeter将安装在 `/usr/local/jmeter` 目录中。

在菜单中选择：

- `1` - 安装JMeter
- `2` - 运行接口测试
- `3` - 运行性能测试

---

## 📋 目录结构

```
jmeter/
├── install-jmeter.sh              # JMeter自动安装脚本
├── run-test.sh                    # 接口测试执行脚本
├── run-performance-test.sh        # 性能测试执行脚本
├── quick-start.sh                 # 快速启动脚本（交互式）
├── configs/
│   └── test-env.properties         # 测试环境配置
├── data/
│   └── test-users.csv             # 测试用户数据
├── test-plans/
│   ├── dehaze-api-test.jmx        # 接口测试计划
│   └── dehaze-performance-test.jmx # 性能测试计划
├── reports/                       # 测试报告目录（自动创建）
├── performance-reports/           # 性能测试报告（自动创建）
├── README.md                      # 详细文档
└── QUICK_START.md                 # 本文件
```

---

## 🔧 安装JMeter

### 方式1: 使用安装脚本（推荐）

```bash
cd /data/workspace/dehaze-system/jmeter
sudo ./install-jmeter.sh

# 使环境变量生效
source /etc/profile.d/jmeter.sh

# 验证安装
jmeter --version
```

### 方式2: 使用快速启动脚本

```bash
./quick-start.sh
# 选择 1. 安装JMeter
```

### 手动安装（备选）

```bash
# 下载JMeter
cd /tmp
wget https://downloads.apache.org//jmeter/binaries/apache-jmeter-5.6.3.tgz

# 解压到指定目录
sudo mkdir -p /usr/local/jmeter
sudo tar -xzf apache-jmeter-5.6.3.tgz -C /usr/local/jmeter

# 配置环境变量
echo 'export JMETER_HOME=/usr/local/jmeter/apache-jmeter-5.6.3' | sudo tee -a /etc/profile.d/jmeter.sh
echo 'export PATH=$JMETER_HOME/bin:$PATH' | sudo tee -a /etc/profile.d/jmeter.sh

# 使环境变量生效
source /etc/profile.d/jmeter.sh
```

---

## 🧪 运行测试

### 1. 配置测试环境

编辑 `configs/test-env.properties`：

```properties
# 修改服务器地址
TEST_BASE_URL=http://your-server:8080

# 修改测试账号
TEST_USERNAME=admin
TEST_PASSWORD=your_password
```

### 2. 准备测试数据

```bash
# 创建测试文件
echo "Test content" > /tmp/test-file.txt
```

### 3. 运行接口测试

#### 方式1: 使用脚本（推荐）

```bash
cd /data/workspace/dehaze-system/jmeter
./run-test.sh
```

#### 方式2: 使用JMeter命令

```bash
# 命令行模式（无GUI）
jmeter -n -t test-plans/dehaze-api-test.jmx \
       -l reports/result-$(date +%Y%m%d-%H%M%S).jtl \
       -e -o reports/html-report-$(date +%Y%m%d-%H%M%S)

# GUI模式（用于调试）
jmeter -t test-plans/dehaze-api-test.jmx
```

#### 方式3: 使用快速启动脚本

```bash
./quick-start.sh
# 选择 2. 运行接口测试
```

### 4. 运行性能测试

#### 使用脚本（推荐）

```bash
cd /data/workspace/dehaze-system/jmeter

# 默认参数：10个并发用户，循环100次，启动时间10秒
./run-performance-test.sh

# 自定义参数：50个并发用户，循环200次，启动时间30秒
./run-performance-test.sh 50 200 30

# 查看帮助
./run-performance-test.sh --help
```

#### 使用快速启动脚本

```bash
./quick-start.sh
# 选择 3. 运行性能测试
# 然后输入测试参数
```

---

## 📊 查看测试报告

### 命令行查看

```bash
# 查看最新的报告
ls -lt reports/html-report-* | head -1

# 查看性能测试报告
ls -lt performance-reports/html-report-* | head -1
```

### 浏览器查看

```bash
# 使用快速启动脚本
./quick-start.sh
# 选择 5. 查看测试报告

# 或直接打开
xdg-open reports/html-report-xxxxxx/index.html
```

### 报告内容

测试报告包含：

| 报告类型       | 说明                     |
|------------|------------------------|
| Dashboard  | 测试概览、平均响应时间、吞吐量、错误率    |
| Charts     | 响应时间趋势图、活跃线程趋势图、吞吐量趋势图 |
| Statistics | 请求总数、成功/失败数量、百分位响应时间   |
| Errors     | 失败的请求列表、错误原因分析         |

---

## 🛠️ 常用命令

### 环境检查

```bash
# 检查Java版本
java -version

# 检查JMeter版本
jmeter --version

# 健康检查
./quick-start.sh --check

# 或
./quick-start.sh
# 选择 6. 环境健康检查
```

### 测试相关

```bash
# 仅检查环境，不运行测试
./run-test.sh --no-report

# GUI模式调试测试计划
jmeter -t test-plans/dehaze-api-test.jmx

# 清理旧的测试报告
rm -rf reports/html-report-*
rm -rf performance-reports/html-report-*
```

---

## 🐛 故障排查

### 问题1: JMeter未找到

**错误信息**: `jmeter: command not found`

**解决方案**:

```bash
# 检查JMeter是否安装
ls -la /opt/jmeter/current/bin/jmeter

# 配置环境变量
source /etc/profile.d/jmeter.sh

# 验证
jmeter --version
```

### 问题2: 连接失败

**错误信息**: 所有请求显示连接失败

**解决方案**:

```bash
# 检查服务是否运行
curl http://localhost:8080/actuator/health

# 检查端口是否开放
nc -z localhost 8080

# 检查配置文件
cat configs/test-env.properties

# 修改服务器地址
vim configs/test-env.properties
```

### 问题3: 认证失败

**错误信息**: 登录接口返回401/403

**解决方案**:

```bash
# 1. 检查用户名密码
cat configs/test-env.properties

# 2. 检查后端日志
# dehaze-java 应用的日志输出

# 3. 使用正确的测试账号
TEST_USERNAME=admin
TEST_PASSWORD=123456
```

### 问题4: 内存不足

**错误信息**: `OutOfMemoryError`

**解决方案**:

```bash
# 修改JMeter堆内存配置
vim $JMETER_HOME/bin/jmeter

# 修改HEAP参数（增加到4GB）
HEAP="-Xms2g -Xmx4g -XMaxMetaspaceSize:512m"

# 重新运行测试
./run-test.sh
```

### 问题5: 测试文件不存在

**错误信息**: 文件上传测试失败

**解决方案**:

```bash
# 创建测试文件
echo "Test content" > /tmp/test-file.txt

# 验证文件存在
ls -la /tmp/test-file.txt
```

---

## 📈 性能测试建议

### 测试参数选择

| 场景   | 并发用户数 | 循环次数 | 启动时间 | 总请求数   |
|------|-------|------|------|--------|
| 轻量测试 | 10    | 100  | 10   | 1000   |
| 中等负载 | 50    | 200  | 30   | 10000  |
| 高负载  | 100   | 500  | 60   | 50000  |
| 极限压力 | 200   | 1000 | 120  | 200000 |

### 性能指标参考

| 指标      | 优秀          | 良好             | 需优化        |
|---------|-------------|----------------|------------|
| 成功率     | ≥99%        | 95-99%         | <95%       |
| 平均响应时间  | <200ms      | 200-500ms      | >500ms     |
| 95%响应时间 | <500ms      | 500-1000ms     | >1000ms    |
| 吞吐量     | >1000 req/s | 500-1000 req/s | <500 req/s |

---

## 🔍 测试用例说明

### 认证接口测试

- ✅ 获取验证码（GET `/api/v1/auth/captcha`）
- ✅ 用户登录（POST `/api/v1/auth/login`）
- ✅ 用户注销（DELETE `/api/v1/auth/logout`）

### 文件管理接口测试

- ✅ 分页查询文件（GET `/api/v1/files/page`）
- ✅ 文件上传（POST `/api/v1/files`）
- ✅ 获取文件详情（GET `/api/v1/files/{fileId}`）
- ✅ 文件校验（GET `/api/v1/files/check`）
- ✅ 删除文件（DELETE `/api/v1/files`）
- ✅ 下载文件（GET `/api/v1/files/download/{taskId}`）

---

## 📚 进阶使用

### 1. 修改测试计划

使用GUI模式编辑测试计划：

```bash
jmeter -t test-plans/dehaze-api-test.jmx
```

在GUI中可以：

- 添加新的测试用例
- 修改断言规则
- 调整测试参数
- 添加定时器、监听器等

### 2. 使用CSV数据文件

编辑 `data/test-users.csv` 添加更多测试用户：

```csv
username,password
admin,123456
test001,test123
test002,test123
```

### 3. 集成到CI/CD

在Jenkins/GitLab CI中执行测试：

```bash
#!/bin/bash

# 启动dehaze-java服务
./start-app.sh

# 等待服务启动
sleep 30

# 运行测试
cd /data/workspace/dehaze-system/jmeter
./run-test.sh

# 检查测试结果
if [ $? -eq 0 ]; then
    echo "✓ 测试通过"
    exit 0
else
    echo "✗ 测试失败"
    exit 1
fi
```

---

## 📞 获取帮助

### 查看帮助信息

```bash
# 快速启动脚本帮助
./quick-start.sh --help

# 接口测试脚本帮助
./run-test.sh --help

# 性能测试脚本帮助
./run-performance-test.sh --help
```

### 详细文档

```bash
# 查看详细文档
cat README.md
```

---

## ✅ 检查清单

运行测试前确认：

- [ ] Java已安装（`java -version`）
- [ ] JMeter已安装（`jmeter --version`）
- [ ] dehaze-java服务已启动（端口8080）
- [ ] 测试配置已更新（`configs/test-env.properties`）
- [ ] 测试数据已准备（`/tmp/test-file.txt`）
- [ ] 脚本有执行权限（`chmod +x *.sh`）

---

## 🎯 总结

### 最快上手方式（3步）

```bash
# 1. 安装JMeter
cd /data/workspace/dehaze-system/jmeter
sudo ./install-jmeter.sh
source /etc/profile.d/jmeter.sh

# 2. 运行测试
./quick-start.sh
# 选择 2. 运行接口测试

# 3. 查看报告
./quick-start.sh
# 选择 5. 查看测试报告
```

### 推荐工作流

1. 开发完成后运行接口测试验证功能
2. 上线前运行性能测试评估系统容量
3. 定期运行测试监控系统稳定性
4. 将测试集成到CI/CD流程中

---

**祝测试顺利！** 🎉
