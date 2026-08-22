import request from "@/utils/request";
import {
  AnalyzeRequest,
  RecommendationFeedback,
  RecommendationReport,
  RecommendationResult,
  RecommendationRule,
} from "./model";

/**
 * 推荐管理 API
 *
 * 根据图像特征、用户偏好、算法表现等多维信息，为用户推荐最合适的去雾算法。
 * 接口契约参见：03-模块设计/基础模块/推荐管理/API接口.md
 *
 * - analyze / algorithms / feedback：仅需登录用户身份，无特殊权限标识
 * - rules / report：管理员接口，需 sys:recommendation:rule:view / edit / report 权限
 *
 * 图像特征分析为同步返回（性能目标 < 2s），无需轮询。
 */
class RecommendationAPI {
  /**
   * 图像特征分析（F-REC-001）
   * POST /api/v1/recommendations/analyze
   *
   * 上传图片或指定 imageId，返回 7 维特征分析结果。
   * 业务错误码：A0401(图片不存在)、A0701(格式不支持)、A0702(超限)、B0100(分析超时)
   */
  static analyze(data: AnalyzeRequest) {
    return request<RecommendationResult["analysis"]>({
      url: "/api/v1/recommendations/analyze",
      method: "post",
      data,
    });
  }

  /**
   * 获取算法推荐（F-REC-002）
   * GET /api/v1/recommendations/algorithms
   *
   * 基于分析结果返回 Top 3 推荐算法及匹配度和理由。
   */
  static getAlgorithmRecommendations(params: { analysisId?: number; imageMd5?: string }) {
    return request<RecommendationResult["recommendations"]>({
      url: "/api/v1/recommendations/algorithms",
      method: "get",
      params,
    });
  }

  /**
   * 提交推荐反馈（F-REC-003）
   * POST /api/v1/recommendations/feedback
   *
   * 用户对推荐结果进行有用/无用反馈，反馈数据用于优化推荐模型。
   */
  static submitFeedback(data: RecommendationFeedback) {
    return request<{ id: number }>({
      url: "/api/v1/recommendations/feedback",
      method: "post",
      data,
    });
  }

  /**
   * 获取推荐规则配置（管理员，F-REC-004）
   * GET /api/v1/recommendations/rules
   *
   * 权限标识：sys:recommendation:rule:view
   */
  static getRules() {
    return request<RecommendationRule[]>({
      url: "/api/v1/recommendations/rules",
      method: "get",
    });
  }

  /**
   * 更新推荐规则配置（管理员，F-REC-004）
   * PUT /api/v1/recommendations/rules
   *
   * 权限标识：sys:recommendation:rule:edit
   * 业务错误码：A0500(规则格式不合法)、A0502(修改被禁用的规则)
   */
  static updateRule(id: number, data: RecommendationRule) {
    return request<number>({
      url: "/api/v1/recommendations/rules",
      method: "put",
      // 后端契约：id 随请求体提交（body.id 为 0 时走新增，否则更新）
      data: { ...data, id },
    });
  }

  /**
   * 推荐效果报表（管理员，F-REC-004）
   * GET /api/v1/recommendations/report
   *
   * 权限标识：sys:recommendation:report
   */
  static getReport(params?: { startDate?: string; endDate?: string }) {
    return request<RecommendationReport>({
      url: "/api/v1/recommendations/report",
      method: "get",
      params,
    });
  }
}

export default RecommendationAPI;
