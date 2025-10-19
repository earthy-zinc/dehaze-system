import { initService } from 'dehaze-sdk-js';

// Clear all mocks before each test
beforeEach(() => {
  jest.clearAllMocks();
});

// 初始化服务以避免网络请求
initService({
  baseURL: 'http://localhost:8989',
  timeout: 5000,
});