import { pageQuery } from "./common";
import type {
  BillingRecordQuery,
  BillingRefundApplyForm,
  BillingStatQuery,
  CreditAdjustForm,
  CreditLogQuery,
} from "../../src/api/ai-billing/model";

/** 计费明细查询参数工厂 */
export const createBillingRecordQuery = (overrides?: Partial<BillingRecordQuery>) =>
  pageQuery<BillingRecordQuery>({
    ...overrides,
  });

/** 流水查询参数工厂 */
export const createCreditLogQuery = (overrides?: Partial<CreditLogQuery>) =>
  pageQuery<CreditLogQuery>({
    ...overrides,
  });

/** 统计查询参数工厂 */
export const createBillingStatQuery = (
  overrides?: Partial<BillingStatQuery>
): BillingStatQuery => ({
  groupBy: "model",
  ...overrides,
});

/** 退款申请表单工厂 */
export const createRefundApplyForm = (
  overrides?: Partial<BillingRefundApplyForm>
): BillingRefundApplyForm => ({
  billingId: 1,
  amount: 100,
  reason: "测试误扣退款",
  ...overrides,
});

/** 管理员调整积分表单工厂 */
export const createCreditAdjustForm = (
  overrides?: Partial<CreditAdjustForm>
): CreditAdjustForm => ({
  userId: 3,
  amount: 100,
  reason: "测试手动调整",
  ...overrides,
});
