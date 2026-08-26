import dotenv from "dotenv";
import path from "path";

// 加载项目根 .env 到 process.env
dotenv.config({
  path: path.resolve(__dirname, "../../../.env"),
  quiet: true,
});

// 登录种子账号 admin 的密码（bcrypt 固化在种子 SQL，此变量仅作登录凭证声明）
export const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || "Dehaze2026";

export const MYSQL_HOST = process.env.MYSQL_HOST || "127.0.0.1";
export const MYSQL_PORT = Number(process.env.MYSQL_PORT || "3306");
export const MYSQL_USERNAME = process.env.MYSQL_USERNAME || "root";
export const MYSQL_PASSWORD = process.env.MYSQL_PASSWORD || "Dehaze2026";
export const MYSQL_DATABASE = process.env.MYSQL_DATABASE || "dehaze";

export const REDIS_HOST = process.env.REDIS_HOST || "127.0.0.1";
export const REDIS_PORT = Number(process.env.REDIS_PORT || "6379");
export const REDIS_PASSWORD = process.env.REDIS_PASSWORD || "Dehaze2026";
export const REDIS_DATABASE = Number(process.env.REDIS_DATABASE || "0");

export const NGINX_STATIC_HOST = process.env.NGINX_STATIC_HOST || "127.0.0.1";
export const NGINX_STATIC_PORT = process.env.NGINX_STATIC_PORT || "9000";

export const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8989";

const PORT_TO_BACKEND: Record<string, string> = {
  "8989": "dehaze-java",
  "8990": "dehaze-go",
  "8991": "dehaze-python",
};
export const BACKEND_NAME = PORT_TO_BACKEND[new URL(BACKEND_URL).port] || "dehaze-java";
