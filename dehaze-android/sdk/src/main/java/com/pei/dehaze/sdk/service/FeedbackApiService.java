package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.feedback.FeedbackCloseForm;
import com.pei.dehaze.sdk.model.feedback.FeedbackCreateForm;
import com.pei.dehaze.sdk.model.feedback.FeedbackDetailVO;
import com.pei.dehaze.sdk.model.feedback.FeedbackPageVO;
import com.pei.dehaze.sdk.model.feedback.FeedbackReplyForm;
import com.pei.dehaze.sdk.model.feedback.MyRatingVO;
import com.pei.dehaze.sdk.model.feedback.RatingCreateForm;
import com.pei.dehaze.sdk.model.feedback.RatingDetailVO;
import com.pei.dehaze.sdk.model.feedback.RatingPageVO;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.PUT;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 反馈评价管理API服务接口
 * 对齐后端路由：/api/v1/feedback
 */
public interface FeedbackApiService {

    // ===== 评价 =====

    @POST("/api/v1/feedback/ratings")
    Call<Result<Object>> createRating(@Body RatingCreateForm form);

    @PUT("/api/v1/feedback/ratings/{id}")
    Call<Result<Void>> updateRating(@Path("id") long id, @Body RatingCreateForm form);

    @GET("/api/v1/feedback/ratings/my")
    Call<Result<PageResult<MyRatingVO>>> listMyRatings(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize);

    @GET("/api/v1/feedback/ratings/by-prediction/{predictionLogId}")
    Call<Result<RatingDetailVO>> getRatingByPrediction(@Path("predictionLogId") long predictionLogId);

    @GET("/api/v1/feedback/ratings/page")
    Call<Result<PageResult<RatingPageVO>>> listRatings(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("keywords") String keywords,
            @Query("algorithmId") Long algorithmId,
            @Query("ratingMin") Integer ratingMin,
            @Query("ratingMax") Integer ratingMax,
            @Query("hasComment") Boolean hasComment,
            @Query("startTime") String startTime,
            @Query("endTime") String endTime);

    @PUT("/api/v1/feedback/ratings/{id}/hide")
    Call<Result<Void>> hideRating(@Path("id") long id);

    @POST("/api/v1/feedback/ratings/{id}/reply")
    Call<Result<Void>> replyRating(@Path("id") long id, @Body Object content);

    // ===== 反馈 =====

    @POST("/api/v1/feedback")
    Call<Result<Object>> createFeedback(@Body FeedbackCreateForm form);

    @GET("/api/v1/feedback/my")
    Call<Result<PageResult<FeedbackPageVO>>> listMyFeedback(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize);

    @GET("/api/v1/feedback/{id}")
    Call<Result<FeedbackDetailVO>> getFeedbackDetail(@Path("id") long id);

    @POST("/api/v1/feedback/{id}/supplement")
    Call<Result<Void>> supplementFeedback(@Path("id") long id, @Body Object form);

    @GET("/api/v1/feedback/page")
    Call<Result<PageResult<FeedbackPageVO>>> listFeedback(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("keywords") String keywords,
            @Query("feedbackType") String feedbackType,
            @Query("status") String status,
            @Query("relatedModule") String relatedModule,
            @Query("priority") Integer priority,
            @Query("assigneeId") Long assigneeId,
            @Query("startTime") String startTime,
            @Query("endTime") String endTime);

    @PUT("/api/v1/feedback/{id}/assign")
    Call<Result<Void>> assignFeedback(@Path("id") long id, @Body Object form);

    @POST("/api/v1/feedback/{id}/reply")
    Call<Result<Void>> replyFeedback(@Path("id") long id, @Body FeedbackReplyForm form);

    @PUT("/api/v1/feedback/{id}/close")
    Call<Result<Void>> closeFeedback(@Path("id") long id, @Body FeedbackCloseForm form);

    @PUT("/api/v1/feedback/{id}/tags")
    Call<Result<Void>> updateFeedbackTags(@Path("id") long id, @Body String[] tags);
}
