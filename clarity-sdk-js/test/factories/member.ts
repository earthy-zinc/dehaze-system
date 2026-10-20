import {
  BenefitForm,
  MemberGrowthAdjustForm,
  MemberLevelAdjustForm,
  MemberQuery,
} from "@/api/member/model";
import { pageQuery, uniqueName } from "./common";

export function createMemberQuery(overrides: Partial<MemberQuery> = {}): MemberQuery {
  return pageQuery<MemberQuery>({ ...overrides });
}

export function createLevelAdjustForm(
  overrides: Partial<MemberLevelAdjustForm> = {}
): MemberLevelAdjustForm {
  return {
    levelCode: "level_1",
    reason: uniqueName("测试等级调整"),
    ...overrides,
  };
}

export function createGrowthAdjustForm(
  overrides: Partial<MemberGrowthAdjustForm> = {}
): MemberGrowthAdjustForm {
  return {
    changeValue: 100,
    reason: uniqueName("测试成长值调整"),
    ...overrides,
  };
}

export function createBenefitForm(overrides: Partial<BenefitForm> = {}): BenefitForm {
  return {
    monthlyDehazeQuota: 100,
    monthlyEvaluateQuota: 50,
    historyRetention: 30,
    batchLimit: 10,
    priority: 2,
    ...overrides,
  };
}
