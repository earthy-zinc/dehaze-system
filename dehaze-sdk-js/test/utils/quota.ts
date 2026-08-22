import { MemberAPI } from "../../index";
import { login } from "./auth";
import { USERS } from "../factories/constants";

/**
 * 确保普通用户（level_0）的月度去雾配额充足（测试套件自愈）。
 *
 * 全量串行运行时 core-flow/favorite-integration/recommendation-integration/model
 * 多个文件累计消耗 USER 的月度配额（level_0 默认仅 20 次），耗尽后提交预测返回
 * A0515「当月次数已用完」。此处提升 level_0 权益配额并通过等级调整刷新 USER
 * 会员配额快照（后端会同步删除配额 Redis 计数器），保证套件可重复运行。
 * 调用方需在之后重新 login 为目标测试账号。
 */
export async function ensureDehazeQuota(minQuota = 200): Promise<void> {
  await login(USERS.ADMIN.username);
  const benefits = await MemberAPI.listBenefits();
  const level0 = benefits.find((b) => b.levelCode === "level_0");
  if (!level0) {
    return;
  }
  if (level0.monthlyDehazeQuota < minQuota) {
    await MemberAPI.updateBenefit("level_0", {
      monthlyDehazeQuota: minQuota,
      monthlyEvaluateQuota: level0.monthlyEvaluateQuota,
      historyRetention: level0.historyRetention,
      batchLimit: level0.batchLimit,
      priority: level0.priority,
    });
  }
  const detail = await MemberAPI.getDetail(USERS.USER.id);
  if ((detail.monthlyDehazeQuota ?? 0) < minQuota) {
    await MemberAPI.adjustLevel(USERS.USER.id, {
      levelCode: "level_0",
      reason: "测试套件配额自愈",
    });
  }
}
