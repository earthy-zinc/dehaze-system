import dotenv from "dotenv";
import path from "path";

// 加载项目根 .env 到 process.env
dotenv.config({
  path: path.resolve(__dirname, "../../../.env"),
  quiet: true,
});

// 种子账号（admin/root/test 等）口令明文，对应 config/sql/data/sys_user.sql 的 bcrypt；
// 非运行时配置：改 .env 不会改变种子口令，只能改 sql 里的哈希
export const SEED_PASSWORD = "Clarity2026";

// 连接与凭据类变量必须由根 .env 提供：缺失即报错，避免兜底值（127.0.0.1 / 生产预留库名 clarity）静默连到错误环境
function required(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(
      `缺少环境变量 ${name}：请确认仓库根 .env 已按环境模板落地（.env.dev.example / .env.pre.example）`
    );
  }
  return value;
}

export const MYSQL_HOST = required("MYSQL_HOST");
export const MYSQL_PORT = Number(required("MYSQL_PORT"));
export const MYSQL_USERNAME = required("MYSQL_USERNAME");
export const MYSQL_PASSWORD = required("MYSQL_PASSWORD");
export const MYSQL_DATABASE = required("MYSQL_DATABASE");

export const REDIS_HOST = required("REDIS_HOST");
export const REDIS_PORT = Number(required("REDIS_PORT"));
export const REDIS_PASSWORD = required("REDIS_PASSWORD");
export const REDIS_DATABASE = Number(required("REDIS_DATABASE"));

export const NGINX_STATIC_HOST = required("NGINX_STATIC_HOST");
export const NGINX_STATIC_PORT = Number(required("NGINX_STATIC_PORT"));

export const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8989";

const PORT_TO_BACKEND: Record<string, string> = {
  "8989": "clarity-java",
  "8990": "clarity-go",
  "8991": "clarity-python",
};
export const BACKEND_NAME = PORT_TO_BACKEND[new URL(BACKEND_URL).port] || "clarity-java";
