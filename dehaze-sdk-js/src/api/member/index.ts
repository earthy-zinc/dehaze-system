import { PageResult } from "@/types";
import request from "@/utils/request";
import {
  BenefitForm,
  BenefitSummaryVO,
  BenefitVO,
  GrowthLogQuery,
  GrowthLogVO,
  MemberDetailVO,
  MemberGrowthAdjustForm,
  MemberLevelAdjustForm,
  MemberPageVO,
  MemberProfileVO,
  MemberQuery,
  MemberStatusForm,
  SignInCalendarVO,
  SignInResultVO,
  TrialStatusVO,
} from "./model";

class MemberAPI {
  /** 当前用户会员信息 */
  static getProfile() {
    return request<MemberProfileVO>({
      url: "/api/v1/members/profile",
      method: "get",
    });
  }

  /** 成长值变动明细 */
  static getGrowthLogs(queryParams: GrowthLogQuery) {
    return request<PageResult<GrowthLogVO[]>>({
      url: "/api/v1/members/growth-logs",
      method: "get",
      params: queryParams,
    });
  }

  /** 每日签到 */
  static signIn() {
    return request<SignInResultVO>({
      url: "/api/v1/members/sign-in",
      method: "post",
    });
  }

  /** 签到日历 */
  static getSignInCalendar(year: number, month: number) {
    return request<SignInCalendarVO>({
      url: "/api/v1/members/sign-in/calendar",
      method: "get",
      params: { year, month },
    });
  }

  /** 后台：会员分页列表 */
  static getPage(queryParams: MemberQuery) {
    return request<PageResult<MemberPageVO[]>>({
      url: "/api/v1/members/page",
      method: "get",
      params: queryParams,
    });
  }

  /** 后台：会员详情 */
  static getDetail(userId: number) {
    return request<MemberDetailVO>({
      url: "/api/v1/members/" + userId,
      method: "get",
    });
  }

  /** 后台：等级调整 */
  static adjustLevel(userId: number, data: MemberLevelAdjustForm) {
    return request({
      url: "/api/v1/members/" + userId + "/level",
      method: "put",
      data,
    });
  }

  /** 后台：成长值调整 */
  static adjustGrowth(userId: number, data: MemberGrowthAdjustForm) {
    return request({
      url: "/api/v1/members/" + userId + "/growth",
      method: "put",
      data,
    });
  }

  /** 后台：冻结/解冻 */
  static updateStatus(userId: number, data: MemberStatusForm) {
    return request({
      url: "/api/v1/members/" + userId + "/status",
      method: "put",
      data,
    });
  }

  /** 当前用户会员权益汇总（含 AI 功能限额明细） */
  static getBenefitSummary() {
    return request<BenefitSummaryVO>({
      url: "/api/v1/members/benefit-summary",
      method: "get",
    });
  }

  /** 当前用户试用开通状态 */
  static getTrialStatus() {
    return request<TrialStatusVO>({
      url: "/api/v1/members/trial-status",
      method: "get",
    });
  }

  /** 后台：权益配置列表 */
  static listBenefits() {
    return request<BenefitVO[]>({
      url: "/api/v1/members/benefits",
      method: "get",
    });
  }

  /** 后台：修改权益配置 */
  static updateBenefit(levelCode: string, data: BenefitForm) {
    return request({
      url: "/api/v1/members/benefits/" + levelCode,
      method: "put",
      data,
    });
  }
}

export default MemberAPI;
