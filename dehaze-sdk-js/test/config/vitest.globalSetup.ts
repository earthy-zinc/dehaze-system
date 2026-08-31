/**
 * Vitest 全局测试环境配置（仅在所有测试启动前运行一次）
 *
 * 与 setupFiles 的区别：
 * - setupFiles 在每个测试文件执行前都运行一次
 * - globalSetup 在整个测试运行中只运行一次
 *
 * 这里使用 globalSetup 做一次性 Redis 清理：
 * - 清理缓存前缀，避免测试间状态污染
 * - 清理限流计数，避免上一轮测试残留的计数导致本轮触发 429
 *
 * 三端限流 key 前缀：
 * - Java: rate:limit:login:{ip}（@RateLimit 注解，仅登录/注册限流）
 * - Python: rate:limit:{path}:{ip}（RateLimitMiddleware，按路径限流）
 *
 * .env 由 #/config/constant 统一加载（经 #/utils/redis、#/utils/mysql 间接导入）
 */
import { getRedis, disconnectRedis } from "#/utils/redis";
import { resetMemberQuota, disconnectMysql } from "#/utils/mysql";
import { ensureTestResources } from "./ensure-test-resources";

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
  // 缺失测试资源（图片）会自动生成，避免 beforeAll 读文件 ENOENT 导致的失败与级联跳过
  ensureTestResources();

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
