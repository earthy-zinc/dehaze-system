/**
 * Vitest 全局测试环境配置（仅在所有测试启动前运行一次）
 *
 * 与 setupFiles 的区别：
 * - setupFiles 在每个测试文件执行前都运行一次
 * - globalSetup 在整个测试运行中只运行一次
 *
 * 这里使用 globalSetup 做一次性 Redis 清理，避免：
 * 1. 每个测试文件都清理 session:* 导致 sessionStore 内存缓存失效，进而重复登录
 * 2. 每个测试文件都重新登录，触发后端登录限流（Java rate_limit:login:、Python rate:limit:）
 * 3. 测试间状态污染，影响可重复性
 *
 * 三端限流 key 规范：
 * - Java: rate_limit:login:{ip}:{class}#{method}（@RateLimit 注解）
 * - Python: rate:limit:{path}:{ip}（RateLimitMiddleware）
 * - Go: rate:limit:{prefix}:{ip}（ulule/limiter，未挂载到登录路由）
 */
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import Redis from "ioredis";
import { resetMemberQuota, disconnectMysql } from "./test/utils/mysql";

const backend = process.env.TEST_BACKEND || "java";
const envFile = path.resolve(__dirname, `.env.${backend}`);
if (fs.existsSync(envFile)) {
  dotenv.config({ path: envFile });
}

const CACHE_PREFIXES = [
  "captcha_code:",
  "session:",
  "role:perms:",
  "msg:unread:",
  "feedback:daily:",
  "anti_repeat:",
  "member:quota:",
];

const RATE_LIMIT_PATTERNS = ["*rate_limit:login:*", "*rate:limit:*/api/v1/auth/login:*"];

export async function setup() {
  const redis = new Redis({
    host: process.env.REDIS_HOST || "127.0.0.1",
    port: Number(process.env.REDIS_PORT) || 6379,
    password: process.env.REDIS_PASSWORD || "12345678",
    db: Number(process.env.REDIS_DB) || 0,
    maxRetriesPerRequest: 3,
  });
  redis.on("error", () => {});

  try {
    for (const prefix of CACHE_PREFIXES) {
      const keys = await redis.keys(`${prefix}*`);
      if (keys.length > 0) {
        await redis.del(keys);
      }
    }
    for (const pattern of RATE_LIMIT_PATTERNS) {
      const keys = await redis.keys(pattern);
      if (keys.length > 0) {
        await redis.del(keys);
      }
    }
  } catch {
  } finally {
    await redis.quit();
  }

  try {
    await resetMemberQuota([4, 5, 6, 7, 8]);
  } catch {
  } finally {
    await disconnectMysql();
  }
}
