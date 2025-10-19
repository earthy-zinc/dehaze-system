import axios, { CreateAxiosDefaults } from "axios";

// 创建 axios 实例
const createService = (config?: CreateAxiosDefaults) => {
  const service = axios.create(
    config || {
      baseURL: "http://localhost:5000",
      timeout: 5000,
      headers: {
        "Content-Type": "application/json;charset=utf-8",
      },
    }
  );

  return service;
};

// 初始化服务实例
let pythonService = createService();

// 提供重新初始化服务的方法
export const initPythonService = (config?: CreateAxiosDefaults) => {
  pythonService = createService(config);
  return pythonService;
};

// 导出 axios 实例
export default pythonService;
