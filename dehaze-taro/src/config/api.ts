/**
 * API 基础地址配置
 *
 * - H5 开发模式：baseURL 置空，请求走 devServer 代理（见 config/dev.ts），规避浏览器 CORS
 * - H5 生产模式：经 Nginx 反向代理，同样使用相对路径
 * - 小程序端：不存在跨域问题，必须使用绝对地址
 */
const isH5 = process.env.TARO_ENV === "h5";

const JAVA_BASE_URL = isH5 ? "" : "http://localhost:8989";
const PYTHON_BASE_URL = isH5 ? "" : "http://localhost:8991";

export const apiConfig = {
  java: JAVA_BASE_URL,
  python: PYTHON_BASE_URL,
};
