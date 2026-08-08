package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.feedback.FeedbackCloseForm;
import com.pei.dehaze.sdk.model.feedback.FeedbackCreateForm;
import com.pei.dehaze.sdk.model.feedback.FeedbackDetailVO;
import com.pei.dehaze.sdk.model.feedback.FeedbackPageVO;
import com.pei.dehaze.sdk.model.feedback.FeedbackQuery;
import com.pei.dehaze.sdk.model.feedback.FeedbackReplyForm;
import com.pei.dehaze.sdk.model.feedback.MyRatingVO;
import com.pei.dehaze.sdk.model.feedback.RatingCreateForm;
import com.pei.dehaze.sdk.model.feedback.RatingDetailVO;
import com.pei.dehaze.sdk.model.feedback.RatingPageVO;
import com.pei.dehaze.sdk.model.feedback.RatingQuery;

import retrofit2.Call;

/**
 * 反馈评价管理API接口封装
 * 对齐后端路由：/api/v1/feedback
 */
public class FeedbackAPI {

    private FeedbackAPI() {
    }

    // ===== 评价 =====

    public static void createRating(RatingCreateForm form, ApiCallback<Object> callback) {
        Call<Result<Object>> call = DehazeSDK.getInstance().getFeedbackApiService().createRating(form);
        call.enqueue(callback);
    }

    public static void updateRating(long id, RatingCreateForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getFeedbackApiService().updateRating(id, form);
        call.enqueue(callback);
    }

    public static void listMyRatings(int pageNum, int pageSize, ApiCallback<PageResult<MyRatingVO>> callback) {
        Call<Result<PageResult<MyRatingVO>>> call = DehazeSDK.getInstance().getFeedbackApiService().listMyRatings(pageNum, pageSize);
        call.enqueue(callback);
    }

    public static void getRatingByPrediction(long predictionLogId, ApiCallback<RatingDetailVO> callback) {
        Call<Result<RatingDetailVO>> call = DehazeSDK.getInstance().getFeedbackApiService().getRatingByPrediction(predictionLogId);
        call.enqueue(callback);
    }

    public static void listRatings(RatingQuery query, ApiCallback<PageResult<RatingPageVO>> callback) {
        Call<Result<PageResult<RatingPageVO>>> call = DehazeSDK.getInstance().getFeedbackApiService().listRatings(
                query.getPageNum(), query.getPageSize(),
                query.getKeywords(), query.getAlgorithmId(),
                query.getRatingMin(), query.getRatingMax(),
                query.getHasComment(), query.getStartTime(), query.getEndTime());
        call.enqueue(callback);
    }

    public static void hideRating(long id, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getFeedbackApiService().hideRating(id);
        call.enqueue(callback);
    }

    public static void replyRating(long id, String content, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getFeedbackApiService().replyRating(id, new Object() {
            public final String replyContent = content;
        });
        call.enqueue(callback);
    }

    // ===== 反馈 =====

    public static void createFeedback(FeedbackCreateForm form, ApiCallback<Object> callback) {
        Call<Result<Object>> call = DehazeSDK.getInstance().getFeedbackApiService().createFeedback(form);
        call.enqueue(callback);
    }

    public static void listMyFeedback(int pageNum, int pageSize, ApiCallback<PageResult<FeedbackPageVO>> callback) {
        Call<Result<PageResult<FeedbackPageVO>>> call = DehazeSDK.getInstance().getFeedbackApiService().listMyFeedback(pageNum, pageSize);
        call.enqueue(callback);
    }

    public static void getFeedbackDetail(long id, ApiCallback<FeedbackDetailVO> callback) {
        Call<Result<FeedbackDetailVO>> call = DehazeSDK.getInstance().getFeedbackApiService().getFeedbackDetail(id);
        call.enqueue(callback);
    }

    public static void listFeedback(FeedbackQuery query, ApiCallback<PageResult<FeedbackPageVO>> callback) {
        Call<Result<PageResult<FeedbackPageVO>>> call = DehazeSDK.getInstance().getFeedbackApiService().listFeedback(
                query.getPageNum(), query.getPageSize(),
                query.getKeywords(), query.getFeedbackType(), query.getStatus(),
                query.getRelatedModule(), query.getPriority(), query.getAssigneeId(),
                query.getStartTime(), query.getEndTime());
        call.enqueue(callback);
    }

    public static void replyFeedback(long id, FeedbackReplyForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getFeedbackApiService().replyFeedback(id, form);
        call.enqueue(callback);
    }

    public static void closeFeedback(long id, FeedbackCloseForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getFeedbackApiService().closeFeedback(id, form);
        call.enqueue(callback);
    }
}
