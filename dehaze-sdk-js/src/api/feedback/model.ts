import { PageQuery } from "@/types";

/** 反馈状态(pending:待处理;processing:处理中;replied:已回复;closed:已关闭) */
export type FeedbackStatus = "pending" | "processing" | "replied" | "closed";

/** 反馈类型 */
export type FeedbackType = "suggestion" | "bug" | "experience" | "complaint";

/** 评价回复类型 */
export type FeedbackReplyType = "info" | "resolved" | "unsupported" | "dev_transfer";

/** 回复人类型(1:用户;2:管理员) */
export type ReplierType = 1 | 2;

/** 评价创建表单 */
export interface RatingCreateForm {
  predLogId: number;
  rating: number;
  comment?: string;
  tags?: string[];
  imageUrls?: string[];
  isAnonymous?: number;
}

/** 评价查询参数（后台） */
export interface RatingQuery extends PageQuery {
  keywords?: string;
  algorithmId?: number;
  ratingMin?: number;
  ratingMax?: number;
  hasComment?: boolean;
  tags?: string[];
  startTime?: string;
  endTime?: string;
}

/** 评价列表VO（用户端） */
export interface MyRatingVO {
  id: number;
  predLogId: number;
  algorithmName: string;
  rating: number;
  comment?: string;
  tags?: string[];
  imageUrls?: string[];
  isAnonymous: number;
  adminReply?: string;
  replyTime?: string;
  createTime: string;
}

/** 评价列表VO（后台） */
export interface RatingPageVO extends MyRatingVO {
  userId: number;
  username?: string;
  userAvatar?: string;
  isHidden: number;
}

/** 评价详情VO */
export interface RatingDetailVO extends RatingPageVO {
  algorithmId: number;
}

/** 反馈创建表单 */
export interface FeedbackCreateForm {
  feedbackType: FeedbackType;
  title: string;
  content: string;
  contact?: string;
  images?: string[];
  relatedModule?: string;
}

/** 反馈查询参数（后台） */
export interface FeedbackQuery extends PageQuery {
  keywords?: string;
  feedbackType?: FeedbackType;
  status?: FeedbackStatus;
  relatedModule?: string;
  priority?: number;
  assigneeId?: number;
  startTime?: string;
  endTime?: string;
}

/** 反馈列表VO */
export interface FeedbackPageVO {
  id: number;
  userId: number;
  username: string;
  feedbackType: FeedbackType;
  title: string;
  content: string;
  status: FeedbackStatus;
  priority: number;
  assigneeId?: number;
  assigneeName?: string;
  relatedModule?: string;
  tags?: string[];
  createTime: string;
  updateTime?: string;
}

/** 反馈详情VO */
export interface FeedbackDetailVO extends FeedbackPageVO {
  contact?: string;
  images?: string[];
  assignedTime?: string;
  closeReason?: string;
  replies: FeedbackReplyVO[];
}

/** 反馈回复VO */
export interface FeedbackReplyVO {
  id: number;
  feedbackId: number;
  replierId: number;
  replierName: string;
  replierType: ReplierType;
  content: string;
  replyType?: FeedbackReplyType;
  attachments?: string[];
  createTime: string;
}

/** 反馈补充说明表单 */
export interface FeedbackSupplementForm {
  content: string;
  attachments?: string[];
}

/** 反馈回复表单 */
export interface FeedbackReplyForm {
  content: string;
  replyType?: FeedbackReplyType;
  attachments?: string[];
}

/** 反馈分配表单 */
export interface FeedbackAssignForm {
  assigneeId: number;
}

/** 反馈关闭表单 */
export interface FeedbackCloseForm {
  closeReason: string;
}

/** 评价统计VO */
export interface RatingStatsVO {
  totalRatings: number;
  averageRating: number;
  ratingDistribution: Record<number, number>;
  positiveTagRanking: Array<{ tag: string; count: number }>;
  negativeTagRanking: Array<{ tag: string; count: number }>;
  algorithmStats: Array<{
    algorithmId: number;
    algorithmName: string;
    averageRating: number;
    totalRatings: number;
    lowRatingRate: number;
  }>;
}

/** 反馈统计VO */
export interface FeedbackStatsVO {
  totalFeedback: number;
  typeDistribution: Record<FeedbackType, number>;
  moduleDistribution: Array<{ module: string; count: number }>;
  statusDistribution: Record<FeedbackStatus, number>;
  averageResponseTime: number;
  averageCloseTime: number;
  topKeywords: Array<{ keyword: string; count: number }>;
}
