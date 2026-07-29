/**
 * Vitest 全局测试环境配置（仅在所有测试启动前运行一次）
 *
 * 与 setupFiles 的区别：
 * - setupFiles 在每个测试文件执行前都运行一次
 * - globalSetup 在整个测试运行中只运行一次
 *
 * 这里使用 globalSetup 做一次性 Redis 清理，避免：
 * 1. 每个测试文件都清理 session:* 导致 sessionStore 内存缓存失效，进而重复登录
 * 2. 每个测试文件都重新登录，触发后端登录限流
 * 3. 测试间状态污染，影响可重复性
 *
 * 三端限流 key 统一前缀为 rate:limit:：
 * - Java: rate:limit:login:{ip}（@RateLimit 注解，仅登录/注册限流）
 * - Python: rate:limit:{path}:{ip}（RateLimitMiddleware，全局限流）
 * - Go: rate:limit:{prefix}:{ip}（ulule/limiter，已实现但未挂载到路由）
 */
import dotenv from "dotenv";
import path from "path";
import { getRedis, disconnectRedis } from "#/utils/redis";
import { resetMemberQuota, disconnectMysql } from "#/utils/mysql";

// quiet: 抑制 dotenv v17 的 injected env 提示日志，保持测试输出为纯 JSON
dotenv.config({ path: path.resolve(__dirname, "../../.env"), quiet: true });

const CACHE_PREFIXES = [
  "captcha_code:",
  "session:",
  "role:perms:",
  "msg:unread:",
  "feedback:daily:",
  "anti_repeat:",
  "member:quota:",
];

// 只删计数子 key，不删 Redisson 配置 key（rate:limit:login:{ip} 本身）
const RATE_LIMIT_PATTERNS = ["{rate:limit:login:*", "rate:limit:/api/v1/*"];

export async function setup() {
  const redis = getRedis();

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
    await disconnectRedis();
  }

  try {
    await resetMemberQuota([4, 5, 6, 7, 8]);
  } catch {
  } finally {
    await disconnectMysql();
  }
}
