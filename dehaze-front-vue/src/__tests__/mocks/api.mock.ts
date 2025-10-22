import { http, HttpResponse } from "msw";

// API 基础 URL
const API_BASE_URL = import.meta.env.VITE_APP_BASE_API || "/api/v1";

// 模拟用户登录响应
export const mockLoginResponse = {
  tokenType: "Bearer",
  accessToken:
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwibmFtZSI6IlRlc3QgVXNlciIsImlhdCI6MTUxNjIzOTAyMn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
};

// 模拟用户信息响应
export const mockUserInfo = {
  userId: 1,
  username: "admin",
  nickname: "管理员",
  avatar: "https://via.placeholder.com/150",
  roles: ["ADMIN"],
  perms: ["sys:user:add", "sys:user:edit", "sys:user:delete", "sys:user:view"],
};

// 模拟图片上传响应
export const mockImageUploadResponse = {
  id: 1,
  name: "test-image.jpg",
  url: "https://via.placeholder.com/800x600",
  md5: "d41d8cd98f00b204e9800998ecf8427e",
  size: 102400,
  width: 800,
  height: 600,
};

// 模拟算法列表响应
export const mockAlgorithmList = {
  data: [
    {
      id: 1,
      name: "RIDCP",
      displayName: "RIDCP 去雾算法",
      description: "基于深度学习的图像去雾算法",
      category: "深度学习",
      status: "active",
    },
    {
      id: 2,
      name: "DCP",
      displayName: "暗通道先验",
      description: "经典的图像去雾算法",
      category: "传统算法",
      status: "active",
    },
  ],
  total: 2,
};

// 模拟去雾任务响应
export const mockDehazeTask = {
  taskId: "task-123",
  status: "processing",
  progress: 0,
  originalImageUrl: "https://via.placeholder.com/800x600",
  resultImageUrl: null,
  algorithmName: "RIDCP",
  createdAt: new Date().toISOString(),
};

// MSW 请求处理器
export const apiHandlers = [
  // 登录
  http.post(`${API_BASE_URL}/auth/login`, () => {
    return HttpResponse.json({
      code: 200,
      data: mockLoginResponse,
      msg: "登录成功",
    });
  }),

  // 获取用户信息
  http.get(`${API_BASE_URL}/users/me`, () => {
    return HttpResponse.json({
      code: 200,
      data: mockUserInfo,
      msg: "获取成功",
    });
  }),

  // 登出
  http.post(`${API_BASE_URL}/auth/logout`, () => {
    return HttpResponse.json({
      code: 200,
      data: null,
      msg: "登出成功",
    });
  }),

  // 上传图片
  http.post(`${API_BASE_URL}/images/upload`, () => {
    return HttpResponse.json({
      code: 200,
      data: mockImageUploadResponse,
      msg: "上传成功",
    });
  }),

  // 获取算法列表
  http.get(`${API_BASE_URL}/algorithms`, () => {
    return HttpResponse.json({
      code: 200,
      data: mockAlgorithmList,
      msg: "获取成功",
    });
  }),

  // 创建去雾任务
  http.post(`${API_BASE_URL}/tasks/dehaze`, () => {
    return HttpResponse.json({
      code: 200,
      data: mockDehazeTask,
      msg: "任务创建成功",
    });
  }),

  // 获取任务状态
  http.get(`${API_BASE_URL}/tasks/:taskId`, ({ params }) => {
    return HttpResponse.json({
      code: 200,
      data: {
        ...mockDehazeTask,
        taskId: params.taskId,
        progress: 100,
        status: "completed",
        resultImageUrl: "https://via.placeholder.com/800x600",
      },
      msg: "获取成功",
    });
  }),

  // 获取图片列表
  http.get(`${API_BASE_URL}/images`, () => {
    return HttpResponse.json({
      code: 200,
      data: {
        list: [mockImageUploadResponse],
        total: 1,
      },
      msg: "获取成功",
    });
  }),
];

// 错误响应处理器
export const errorHandlers = [
  // 未授权
  http.get(`${API_BASE_URL}/users/me`, () => {
    return HttpResponse.json(
      {
        code: 401,
        data: null,
        msg: "未授权，请登录",
      },
      { status: 401 }
    );
  }),

  // 登录失败
  http.post(`${API_BASE_URL}/auth/login`, () => {
    return HttpResponse.json(
      {
        code: 400,
        data: null,
        msg: "用户名或密码错误",
      },
      { status: 400 }
    );
  }),
];
