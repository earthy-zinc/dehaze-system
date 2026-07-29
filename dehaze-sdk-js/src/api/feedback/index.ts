import { PageResult } from "@/types";
import request from "@/utils/request";
import {
  FeedbackAssignForm,
  FeedbackCloseForm,
  FeedbackCreateForm,
  FeedbackDetailVO,
  FeedbackPageVO,
  FeedbackQuery,
  FeedbackReplyForm,
  FeedbackStatsVO,
  FeedbackSupplementForm,
  MyRatingVO,
  RatingCreateForm,
  RatingDetailVO,
  RatingPageVO,
  RatingQuery,
  RatingStatsVO,
} from "./model";

class FeedbackAPI {
  // ============ 评价接口 ============

  /** 用户端：提交评分 */
  static createRating(data: RatingCreateForm) {
    return request<{ id: number }>({
      url: "/api/v1/feedback/ratings",
      method: "post",
      data,
    });
  }

  /** 用户端：修改评分 */
  static updateRating(id: number, data: RatingCreateForm) {
    return request({
      url: "/api/v1/feedback/ratings/" + id,
      method: "put",
      data,
    });
  }

  /** 用户端：我的评价列表 */
  static listMyRatings(queryParams: { pageNum?: number; pageSize?: number }) {
    return request<PageResult<MyRatingVO[]>>({
      url: "/api/v1/feedback/ratings/my",
      method: "get",
      params: queryParams,
    });
  }

  /** 用户端：按处理记录查评价 */
  static getRatingByPrediction(predictionLogId: number) {
    return request<RatingDetailVO | undefined>({
      url: "/api/v1/feedback/ratings/by-prediction/" + predictionLogId,
      method: "get",
    });
  }

  /** 后台：评价分页列表 */
  static listRatings(queryParams: RatingQuery) {
    return request<PageResult<RatingPageVO[]>>({
      url: "/api/v1/feedback/ratings/page",
      method: "get",
      params: queryParams,
    });
  }

  /** 后台：隐藏评价 */
  static hideRating(id: number) {
    return request({
      url: `/api/v1/feedback/ratings/${id}/hide`,
      method: "put",
    });
  }

  /** 后台：回复评价 */
  static replyRating(id: number, content: string) {
    return request({
      url: `/api/v1/feedback/ratings/${id}/reply`,
      method: "post",
      data: { content },
    });
  }

  /** 后台：评价统计 */
  static getRatingStats(startTime?: string, endTime?: string) {
    return request<RatingStatsVO>({
      url: "/api/v1/feedback/ratings/stats",
      method: "get",
      params: { startTime, endTime },
    });
  }

  // ============ 反馈接口 ============

  /** 用户端：提交反馈 */
  static createFeedback(data: FeedbackCreateForm) {
    return request<{ id: number }>({
      url: "/api/v1/feedback",
      method: "post",
      data,
    });
  }

  /** 用户端：我的反馈列表 */
  static listMyFeedback(queryParams: { pageNum?: number; pageSize?: number }) {
    return request<PageResult<FeedbackPageVO[]>>({
      url: "/api/v1/feedback/my",
      method: "get",
      params: queryParams,
    });
  }

  /** 用户端/后台：反馈详情 */
  static getFeedbackDetail(id: number) {
    return request<FeedbackDetailVO>({
      url: "/api/v1/feedback/" + id,
      method: "get",
    });
  }

  /** 用户端：补充说明 */
  static supplementFeedback(id: number, data: FeedbackSupplementForm) {
    return request({
      url: `/api/v1/feedback/${id}/supplement`,
      method: "post",
      data,
    });
  }

  /** 后台：反馈分页列表 */
  static listFeedback(queryParams: FeedbackQuery) {
    return request<PageResult<FeedbackPageVO[]>>({
      url: "/api/v1/feedback/page",
      method: "get",
      params: queryParams,
    });
  }

  /** 后台：分配处理人 */
  static assignFeedback(id: number, data: FeedbackAssignForm) {
    return request({
      url: `/api/v1/feedback/${id}/assign`,
      method: "put",
      data,
    });
  }

  /** 后台：回复反馈 */
  static replyFeedback(id: number, data: FeedbackReplyForm) {
    return request({
      url: `/api/v1/feedback/${id}/reply`,
      method: "post",
      data,
    });
  }

  /** 后台：关闭反馈 */
  static closeFeedback(id: number, data: FeedbackCloseForm) {
    return request({
      url: `/api/v1/feedback/${id}/close`,
      method: "put",
      data,
    });
  }

  /** 后台：设置反馈标签 */
  static updateFeedbackTags(id: number, tags: string[]) {
    return request({
      url: `/api/v1/feedback/${id}/tags`,
      method: "put",
      data: tags,
    });
  }

  /** 后台：反馈统计 */
  static getFeedbackStats(startTime?: string, endTime?: string) {
    return request<FeedbackStatsVO>({
      url: "/api/v1/feedback/stats",
      method: "get",
      params: { startTime, endTime },
    });
  }
}

export default FeedbackAPI;
