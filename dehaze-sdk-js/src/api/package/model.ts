import { PageQuery } from "@/types";

/** 会员等级（套餐关联） */
export type PackageLevelCode = "level_1" | "level_2" | "level_3";

/** 计费周期 */
export type PackagePeriod = "monthly" | "quarterly" | "yearly";

/** 套餐状态(1:在售;0:下架) */
export type PackageStatus = 0 | 1;

/** 商品类型(vip:会员卡;credit:积分卡;创建后不可修改) */
export type PackageType = "vip" | "credit";

/** 优惠券类型 */
export type CouponType = "full_reduction" | "discount" | "no_threshold" | "trial";

/** 优惠券有效期类型 */
export type CouponValidType = "fixed" | "relative";

/** 用户优惠券状态(1:未使用;2:已使用;3:已过期;4:已锁定) */
export type UserCouponStatus = 1 | 2 | 3 | 4;

/** 套餐查询参数 */
export interface PackageQuery extends PageQuery {
  name?: string;
  packageType?: PackageType;
  levelCode?: PackageLevelCode;
  period?: PackagePeriod;
  status?: PackageStatus;
  startTime?: string;
  endTime?: string;
}

/** 权益覆盖项（JSON字段） */
export interface BenefitOverrides {
  monthlyDehazeQuota?: number;
  monthlyEvaluateQuota?: number;
  historyRetention?: number;
  batchLimit?: number;
  priority?: number;
  advancedParams?: number;
  hdExport?: number;
  reportExport?: number;
  batchDownload?: number;
}

/** 套餐表单 */
export interface PackageForm {
  id?: number;
  name: string;
  packageType: PackageType;
  creditAmount?: number;
  levelCode: PackageLevelCode;
  period: PackagePeriod;
  periodDays: number;
  originalPrice: number;
  salePrice: number;
  description?: string;
  benefitOverrides?: BenefitOverrides;
  sort?: number;
  status?: PackageStatus;
}

/** 套餐分页VO */
export interface PackagePageVO {
  id: number;
  name: string;
  packageType: PackageType;
  creditAmount?: number;
  levelCode: PackageLevelCode;
  levelName: string;
  period: PackagePeriod;
  periodDays: number;
  originalPrice: number;
  salePrice: number;
  dailyPrice: number;
  creditUnitPrice?: number;
  salesCount: number;
  status: PackageStatus;
  createTime: string;
}

/** 套餐详情VO（用户端） */
export interface PackageDetailVO {
  id: number;
  name: string;
  packageType: PackageType;
  creditAmount?: number;
  creditUnitPrice?: number;
  levelCode: PackageLevelCode;
  levelName: string;
  period: PackagePeriod;
  periodDays: number;
  originalPrice: number;
  salePrice: number;
  dailyPrice: number;
  description?: string;
  benefits: Record<string, number>;
  activePromotions: PromotionVO[];
  salesCount: number;
}

/** 在售套餐列表项：列表接口不查询促销活动，故无 activePromotions */
export type PackageOnSaleVO = Omit<PackageDetailVO, "activePromotions">;

/** 价格计算结果 */
export interface PriceResult {
  originalPrice: number;
  discountAmount: number;
  couponAmount: number;
  payableAmount: number;
}

/** 促销活动类型 */
export type PromotionType = "discount" | "new_user" | "holiday" | "full_reduction";

/** 促销活动状态(1:启用;0:禁用) */
export type PromotionStatus = 0 | 1;

/** 促销活动查询参数 */
export interface PromotionQuery extends PageQuery {
  name?: string;
  type?: PromotionType;
  status?: PromotionStatus;
  startTime?: string;
  endTime?: string;
}

/** 促销活动表单 */
export interface PromotionForm {
  id?: number;
  name: string;
  type: PromotionType;
  description?: string;
  startTime: string;
  endTime: string;
  activityRules?: Record<string, any>;
  newUserOnly?: number;
  status?: PromotionStatus;
}

/** 促销活动关联套餐表单 */
export interface PromotionPackageForm {
  packageIds: number[];
}

/** 促销活动VO */
export interface PromotionVO {
  id: number;
  name: string;
  type: PromotionType;
  description?: string;
  startTime: string;
  endTime: string;
  activityRules?: Record<string, any>;
  newUserOnly: number;
  status: number;
}

/** 优惠券表单 */
export interface CouponForm {
  id?: number;
  name: string;
  type: CouponType;
  faceValue: number;
  threshold?: number;
  validType: CouponValidType;
  validStart?: string;
  validEnd?: string;
  validDays?: number;
  totalQty: number;
  perUserLimit: number;
  /** 适用商品：商品ID或商品类型（NULL/空表示全部适用） */
  applicableScope?: Array<number | PackageType>;
  status?: number;
}

/** 优惠券查询参数 */
export interface CouponQuery extends PageQuery {
  name?: string;
  type?: CouponType;
  status?: number;
}

/** 优惠券模板VO */
export interface CouponVO {
  id: number;
  name: string;
  type: CouponType;
  faceValue: number;
  threshold?: number;
  validType: CouponValidType;
  validStart?: string;
  validEnd?: string;
  validDays?: number;
  totalQty: number;
  issuedQty: number;
  usedQty: number;
  perUserLimit: number;
  /** 适用商品：商品ID或商品类型（NULL/空表示全部适用） */
  applicableScope?: Array<number | PackageType>;
  status: number;
  createTime: string;
}

/** 用户优惠券VO */
export interface UserCouponVO {
  id: number;
  couponId: number;
  couponName: string;
  type: CouponType;
  faceValue: number;
  threshold?: number;
  status: UserCouponStatus;
  receiveTime: string;
  expireTime?: string;
  usedTime?: string;
  usedOrderId?: number;
  /** 适用商品：商品ID或商品类型（NULL/空表示全部适用） */
  applicableScope?: Array<number | PackageType>;
}

/** 批量发放请求 */
export interface CouponBatchDistributeForm {
  couponId: number;
  targetScope: "all" | "level" | "users";
  levelCodes?: string[];
  userIds?: number[];
}

/** 销售统计VO */
export interface SalesStatsVO {
  totalSales: number;
  totalRevenue: number;
  typeStats: Array<{
    packageType: PackageType;
    packageTypeName: string;
    salesCount: number;
    revenue: number;
  }>;
  packageStats: Array<{
    packageId: number;
    packageName: string;
    salesCount: number;
    revenue: number;
  }>;
  levelStats: Array<{
    levelCode: string;
    levelName: string;
    salesCount: number;
    revenue: number;
  }>;
  periodStats: Array<{
    period: string;
    periodName: string;
    salesCount: number;
    revenue: number;
  }>;
  couponStats: {
    totalIssued: number;
    totalUsed: number;
    usageRate: number;
  };
}
