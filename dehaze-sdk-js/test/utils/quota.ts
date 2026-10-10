import { AiBillingAPI } from "../../index";
import { login } from "./auth";
import { getRedis } from "./redis";
import { USERS } from "../factories/constants";

/**
 * 为 admin 充值 AI 积分，避免 AI 对话/语音用例因余额不足（A0682）被拒。
 *
 * 历史遗留的 `ai:balance:{uid}` Redis 缓存可能是 "0.00" 等非整数字符串，
 * 导致 Redis INCRBY 抛 ResponseError；充值失败时清空该缓存后重试一次，
 * 由 MySQL 重新回填整数值。
 */
export async function topUpAdminCredits(amount = 100000): Promise<void> {
  await login(USERS.ADMIN.username);
  try {
    await AiBillingAPI.adjustCredits({
      userId: USERS.ADMIN.id,
      amount,
      reason: "测试套件积分充值",
    });
  } catch {
    const redis = getRedis();
    await redis.del(`ai:balance:${USERS.ADMIN.id}`, `ai:arrears:${USERS.ADMIN.id}`);
    await AiBillingAPI.adjustCredits({
      userId: USERS.ADMIN.id,
      amount,
      reason: "测试套件积分充值",
    });
  }
}
