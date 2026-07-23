/**
 * 环境配置
 * 模拟器用 10.0.2.2 访问宿主机后端；真机用局域网 IP
 */

const isDev = __DEV__;

// 开发期宿主机地址（Android 模拟器：10.0.2.2；iOS 模拟器：localhost）
const DEV_HOST = '10.0.2.2';

export const API_CONFIG = {
  // Java 主后端：认证/用户/数据集/算法/预测/评估/任务/文件
  JAVA_BASE_URL: isDev
    ? `http://${DEV_HOST}:8989`
    : 'https://api.dehaze.com',
  // Python 辅助后端：算法智能推荐/收藏/对比
  PYTHON_BASE_URL: isDev
    ? `http://${DEV_HOST}:8991`
    : 'https://ai.dehaze.com',
} as const;
