import { PageQuery } from "@/types";

/** 会员等级枚举 */
export type MemberLevelCode = "level_0" | "level_1" | "level_2" | "level_3";

/** 等级来源 */
export type MemberLevelSource = "growth" | "purchase" | "admin";

/** 成长值变动类型 */
export type GrowthChangeType =
  | "dehaze"
  | "evaluate"
  | "rating"
  | "sign_in"
  | "sign_in_bonus"
  | "consume"
  | "refund_deduct"
  | "admin_adjust"
  | "ai_consume";

/** 会员状态(1:正常;0:冻结) */
export type MemberStatus = 0 | 1;

/** 会员查询参数 */
export interface MemberQuery extends PageQuery {
  keywords?: string;
  levelCode?: MemberLevelCode;
  status?: MemberStatus;
  expireTimeStart?: string;
  expireTimeEnd?: string;
  growthMin?: number;
  growthMax?: number;
}

/** 成长值流水查询参数 */
export interface GrowthLogQuery extends PageQuery {
  changeType?: GrowthChangeType;
  startTime?: string;
  endTime?: string;
}

/** 等级调整表单 */
export interface MemberLevelAdjustForm {
  levelCode: MemberLevelCode;
  expireTime?: string;
  reason: string;
}

/** 成长值调整表单 */
export interface MemberGrowthAdjustForm {
  changeValue: number;
  reason: string;
}

/** 会员状态变更表单 */
export interface MemberStatusForm {
  status: MemberStatus;
  reason?: string;
}

/** 权益配置表单 */
export interface BenefitForm {
  levelName?: string;
  growthMin?: number;
  growthMax?: number;
  monthlyDehazeQuota?: number;
  monthlyEvaluateQuota?: number;
  aiCreditsDaily?: number;
  aiCreditsMonthly?: number;
  multimodalLimit?: number;
  vipGiftCredits?: number;
  historyRetention?: number;
  batchLimit?: number;
  priority?: number;
  advancedParams?: number;
  hdExport?: number;
  reportExport?: number;
  batchDownload?: number;
  sort?: number;
  status?: number;
}

/** 权益配置VO */
export interface BenefitVO {
  levelCode: MemberLevelCode;
  levelName: string;
  growthMin: number;
  growthMax: number;
  monthlyDehazeQuota: number;
  monthlyEvaluateQuota: number;
  aiCreditsDaily: number;
  aiCreditsMonthly: number;
  multimodalLimit: number;
  vipGiftCredits: number;
  historyRetention: number;
  batchLimit: number;
  priority: number;
  advancedParams: number;
  hdExport: number;
  reportExport: number;
  batchDownload: number;
  sort: number;
  status: number;
}

/** 会员信息VO（用户端profile） */
export interface MemberProfileVO {
  userId: number;
  username: string;
  nickname: string;
  avatar?: string;
  levelCode: MemberLevelCode;
  levelName: string;
  growthValue: number;
  nextLevelGrowth?: number;
  progressPercent: number;
  expireTime?: string;
  monthlyDehazeQuota: number;
  monthlyDehazeUsed: number;
  monthlyEvaluateQuota: number;
  monthlyEvaluateUsed: number;
  benefits: BenefitVO;
  status: MemberStatus;
}

/** 会员分页VO */
export interface MemberPageVO {
  userId: number;
  username: string;
  nickname: string;
  levelCode: MemberLevelCode;
  levelName: string;
  growthValue: number;
  monthlyUsed: number;
  expireTime?: string;
  status: MemberStatus;
  becomeMemberTime?: string;
}

/** 成长值流水VO */
export interface GrowthLogVO {
  id: number;
  changeType: GrowthChangeType;
  changeValue: number;
  balance: number;
  relatedId?: string;
  reason?: string;
  operatorId?: number;
  createTime: string;
}

/** 签到结果VO */
export interface SignInResultVO {
  signDate: string;
  continuousDays: number;
  growthValue: number;
  bonusGrowth: number;
}

/** 签到日历VO */
export interface SignInCalendarVO {
  signDates: string[];
  continuousDays: number;
  totalDays: number;
}

/** 会员详情VO（后台） */
export interface MemberDetailVO extends MemberProfileVO {
  levelSource: MemberLevelSource;
  totalConsumption: number;
  becomeMemberTime?: string;
  frozenReason?: string;
  frozenTime?: string;
  quotaResetMonth?: number;
}

/** 权益概览-图像任务类型 */
export type BenefitTaskType =
  "dehaze" | "derain" | "desnow" | "lowlight" | "super_resolution" | "denoise" | "inpaint";

/** 权益概览-图像任务明细 */
export interface BenefitTaskDetailVO {
  taskType: BenefitTaskType;
  quota: number;
  used: number;
  remaining: number;
}

/** 权益概览-服务类目聚合 */
export interface BenefitCategoryVO {
  remaining: number;
  details?: BenefitTaskDetailVO[];
}

/** 权益概览VO（当前用户会员权益汇总） */
export interface BenefitSummaryVO {
  imageCategory: BenefitCategoryVO;
  evaluateCategory: BenefitCategoryVO;
  aiCategory: {
    creditsBalance: number;
    todayUsed: number;
    dailyLimit: number;
    monthlyLimit: number;
  };
}

/** 试用开通状态VO */
export interface TrialStatusVO {
  showTrialEntry: boolean;
  trialDays: number;
  trialCredits: number;
  voucherActivated: boolean;
  voucherExpireTime?: string;
  aiTrialCreditsBalance: number;
  newUserExclusiveAvailable: boolean;
  paidMembership: boolean;
}
