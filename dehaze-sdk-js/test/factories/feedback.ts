import {
  FeedbackCreateForm,
  FeedbackQuery,
  FeedbackReplyForm,
  RatingCreateForm,
  RatingQuery,
} from "@/api/feedback/model";
import { pageQuery, uniqueName } from "./common";

const POSITIVE_TAGS = ["去雾彻底", "色彩自然", "细节清晰", "处理速度快", "整体提升明显"];
const NEGATIVE_TAGS = ["残留雾气", "色彩失真", "细节丢失", "处理速度慢", "无明显改善"];

export function createRatingForm(
  predLogId: number,
  overrides: Partial<RatingCreateForm> = {}
): RatingCreateForm {
  return {
    predLogId,
    rating: 5,
    comment: uniqueName("测试评价"),
    tags: [POSITIVE_TAGS[0]!, POSITIVE_TAGS[1]!],
    isAnonymous: 0,
    ...overrides,
  };
}

export function createRatingQuery(overrides: Partial<RatingQuery> = {}): RatingQuery {
  return pageQuery<RatingQuery>({ ...overrides });
}

export function createFeedbackForm(
  overrides: Partial<FeedbackCreateForm> = {}
): FeedbackCreateForm {
  return {
    feedbackType: "suggestion",
    title: uniqueName("测试反馈标题"),
    content: uniqueName("测试反馈内容"),
    contact: "test@example.com",
    ...overrides,
  };
}

export function createFeedbackQuery(overrides: Partial<FeedbackQuery> = {}): FeedbackQuery {
  return pageQuery<FeedbackQuery>({ ...overrides });
}

export function createFeedbackReplyForm(
  overrides: Partial<FeedbackReplyForm> = {}
): FeedbackReplyForm {
  return {
    content: uniqueName("测试回复内容"),
    replyType: "resolved",
    ...overrides,
  };
}

export const RATING_TAGS = {
  POSITIVE: POSITIVE_TAGS,
  NEGATIVE: NEGATIVE_TAGS,
};
