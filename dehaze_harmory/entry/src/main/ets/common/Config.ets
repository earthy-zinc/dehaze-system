/**
 * 全局配置常量
 *
 * 集中管理后端服务地址等环境相关配置，避免在代码中硬编码。
 * 修改部署环境时只需调整此文件。
 */
export class AppConfig {
  /** 后端 Java 服务地址（认证/算法/数据集/预测等核心业务） */
  static readonly BACKEND_BASE_URL: string = 'http://127.0.0.1:8989';

  /** Python 算法服务地址（推理与评估） */
  static readonly PYTHON_BASE_URL: string = 'http://127.0.0.1:8991';

  /** 数据集静态资源服务地址（nginx 托管 hazy/clean 图像对） */
  static readonly DATASET_BASE_URL: string = 'http://127.0.0.1:9000';

  /** 是否开启 SDK 调试日志 */
  static readonly SDK_DEBUG: boolean = true;
}
