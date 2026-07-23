/// <reference types="@tarojs/taro" />

declare module "*.png";
declare module "*.gif";
declare module "*.jpg";
declare module "*.jpeg";
declare module "*.svg";
declare module "*.css";
declare module "*.less";
declare module "*.styl";

declare namespace NodeJS {
  interface ProcessEnv {
    /** NODE 内置环境变量, 会影响到最终构建生成产物 */
    NODE_ENV: "development" | "production";
    /** 当前构建的平台 */
    TARO_ENV:
      | "weapp"
      | "swan"
      | "alipay"
      | "h5"
      | "rn"
      | "tt"
      | "quickapp"
      | "qq"
      | "jd";
    /**
     * 当前构建的小程序 appid
     * @description 若不同环境有不同的小程序，可通过在 env 文件中配置环境变量`TARO_APP_ID`来方便快速切换 appid， 而不必手动去修改 dist/project.config.json 文件
     * @see https://taro-docs.jd.com/docs/next/env-mode-config#特殊环境变量-taro_app_id
     */
    TARO_APP_ID: string;
    /** Java 后端服务地址（小程序端使用，H5 走 devServer 代理） */
    TARO_APP_JAVA_BASE_URL?: string;
    /** Python 后端服务地址（小程序端使用，H5 走 devServer 代理） */
    TARO_APP_PYTHON_BASE_URL?: string;
    /** 数据集静态服务地址（nginx-dataset） */
    TARO_APP_DATASET_BASE_URL?: string;
  }
}
