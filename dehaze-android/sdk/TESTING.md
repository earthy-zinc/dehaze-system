# Dehaze Android SDK 测试指南

## 测试结构

本SDK包含以下类型的测试：

1. **单元测试** - 测试各个组件的基本功能
2. **示例应用** - 演示SDK的使用方法

## 运行测试

### 运行单元测试

```bash
./gradlew test
```

这将运行所有单元测试，并在控制台输出测试结果。

### 运行特定测试类

```bash
./gradlew test --tests "com.pei.dehaze.sdk.DehazeSDKTest"
```

### 查看测试报告

测试报告位于以下目录：
```
build/reports/tests/test/index.html
```

## 测试说明

### DehazeSDKTest
测试SDK的基本功能，包括：
- SDK初始化
- Session 管理
- 认证API调用

### ErrorUtilsTest
测试错误处理工具类的功能：
- 错误解析
- 错误消息生成

### TokenManagerTest
测试 Session 管理器的功能：
- SessionId 的设置和获取
- SessionId 的清除
- Session 失效回调触发

## 编写新测试

1. 在`src/test/java/`目录下创建测试类
2. 使用JUnit 4编写测试方法
3. 使用Mockito进行模拟对象创建
4. 运行测试确保新功能正常工作

## 注意事项

1. 网络测试在没有实际服务器的情况下会失败，这是正常的
2. 测试主要验证API调用的正确性和工具类的功能
3. 实际的API功能需要在有后端服务的情况下才能完全验证