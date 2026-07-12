/**
 * API 基础地址配置
 *
 * H5 开发模式可通过 devServer 代理；小程序必须使用绝对地址。
 */
const JAVA_BASE_URL = 'http://localhost:8989'
const PYTHON_BASE_URL = 'http://localhost:8014'

export const apiConfig = {
  java: JAVA_BASE_URL,
  python: PYTHON_BASE_URL,
}
