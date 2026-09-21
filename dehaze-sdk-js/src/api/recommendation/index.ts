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
 * 根据图像特征分析结果为用户推荐最合适的图像处理算法。
 * 接口契约参见：03-模块设计/基础模块/推荐管理/API接口.md
 *
 * - analyze / algorithms / feedback：仅需登录用户身份，无特殊权限标识
 * - rules / report：管理员接口，需 sys:recommendation:rule:view / edit / report 权限
 */
class RecommendationAPI {
  /**
   * 图像特征分析（F-REC-001）
   * POST /api/v1/recommendations/analyze
   *
   * 上传图片或指定 imageId，返回 7 维特征分析结果。
   * 业务错误码：A0401(imageId方式不支持)、A0400(参数缺失)、A0701(格式不支持)
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
   * 仅允许反馈本人产生的推荐记录，他人记录返回 A0401。
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
   * 后端契约：id 通过 query 参数传递（id=0 表示新增），请求体为规则表单。
   * 业务错误码：A0400(权重/场景类型/算法列表不合法)、A0401(规则不存在)
   */
  static updateRule(id: number, data: RecommendationRule) {
    return request<number>({
      url: "/api/v1/recommendations/rules",
      method: "put",
      params: { id },
      data,
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
