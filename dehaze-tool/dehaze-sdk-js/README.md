# Dehaze SDK for JavaScript

JavaScript SDK for Dehaze System API。详细业务文档见 [dehaze-doc](../../dehaze-doc/docs/05-子项目实现/index.md)。

## 技术栈

- TypeScript
- Axios
- pnpm
- Vitest

## 快速开始

### 安装

```bash
pnpm add dehaze-sdk-js
```

或者使用 npm：

```bash
npm install dehaze-sdk-js
```

### 构建项目

```bash
pnpm run build
```

这将在 `dist/` 目录下生成编译后的文件。

### 清理构建产物

```bash
pnpm run clean
```

### 运行测试

```bash
# 运行所有测试
pnpm run test

# 监听模式运行测试
pnpm run test:watch

# 生成测试覆盖率报告
pnpm run test:coverage

# 运行单个测试
npx vitest run test/user/user.test.ts --testNamePattern="正向测试：获取当前登录用户信息并验证数据完整性"
```

测试使用 Vitest 框架，需要后端服务运行在 `http://localhost:8989`。测试包含数据集、数据项、文件和导出任务的集成测试。

## 许可证

ISC
