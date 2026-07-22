/**
 * API 基础地址配置
 *
 * - H5 开发模式：baseURL 置空，请求走 devServer 代理（见 config/dev.ts），规避浏览器 CORS
 * - H5 生产模式：经 Nginx 反向代理，同样使用相对路径
 * - 小程序端：不存在跨域问题，必须使用绝对地址
 *
 * 后端地址可通过环境变量 TARO_APP_JAVA_BASE_URL / TARO_APP_PYTHON_BASE_URL 覆盖
 * 数据集地址可通过环境变量 TARO_APP_DATASET_BASE_URL 覆盖
 */
const isH5 = process.env.TARO_ENV === "h5";

const JAVA_BASE_URL = isH5
  ? ""
  : process.env.TARO_APP_JAVA_BASE_URL || "http://localhost:8989";
const PYTHON_BASE_URL = isH5
  ? ""
  : process.env.TARO_APP_PYTHON_BASE_URL || "http://localhost:8991";
// 数据集静态服务（nginx-dataset），H5 的 <img> 标签无 CORS 限制，所有端均使用绝对地址
const DATASET_BASE_URL =
  process.env.TARO_APP_DATASET_BASE_URL || "http://localhost:9000";

export const apiConfig = {
  java: JAVA_BASE_URL,
  python: PYTHON_BASE_URL,
  dataset: DATASET_BASE_URL,
};
