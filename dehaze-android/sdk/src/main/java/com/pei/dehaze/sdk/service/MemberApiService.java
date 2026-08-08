package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.member.BenefitForm;
import com.pei.dehaze.sdk.model.member.BenefitVO;
import com.pei.dehaze.sdk.model.member.GrowthAdjustForm;
import com.pei.dehaze.sdk.model.member.GrowthLogQuery;
import com.pei.dehaze.sdk.model.member.GrowthLogVO;
import com.pei.dehaze.sdk.model.member.LevelAdjustForm;
import com.pei.dehaze.sdk.model.member.MemberDetailVO;
import com.pei.dehaze.sdk.model.member.MemberPageVO;
import com.pei.dehaze.sdk.model.member.MemberProfileVO;
import com.pei.dehaze.sdk.model.member.MemberStatusForm;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.PUT;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 会员管理API服务接口
 * 对齐后端路由：/api/v1/members
 */
public interface MemberApiService {

    @GET("/api/v1/members/profile")
    Call<Result<MemberProfileVO>> getProfile();

    @GET("/api/v1/members/growth-logs")
    Call<Result<PageResult<GrowthLogVO>>> getGrowthLogs(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("changeType") String changeType,
            @Query("startTime") String startTime,
            @Query("endTime") String endTime);

    @GET("/api/v1/members/page")
    Call<Result<PageResult<MemberPageVO>>> getPage(
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize,
            @Query("keywords") String keywords,
            @Query("levelCode") String levelCode,
            @Query("status") Integer status,
            @Query("expireTimeStart") String expireTimeStart,
            @Query("expireTimeEnd") String expireTimeEnd,
            @Query("growthMin") Integer growthMin,
            @Query("growthMax") Integer growthMax);

    @GET("/api/v1/members/{userId}")
    Call<Result<MemberDetailVO>> getDetail(@Path("userId") long userId);

    @PUT("/api/v1/members/{userId}/level")
    Call<Result<Void>> adjustLevel(@Path("userId") long userId, @Body LevelAdjustForm form);

    @PUT("/api/v1/members/{userId}/growth")
    Call<Result<Void>> adjustGrowth(@Path("userId") long userId, @Body GrowthAdjustForm form);

    @PUT("/api/v1/members/{userId}/status")
    Call<Result<Void>> updateStatus(@Path("userId") long userId, @Body MemberStatusForm form);

    @GET("/api/v1/members/benefits")
    Call<Result<List<BenefitVO>>> listBenefits();

    @PUT("/api/v1/members/benefits/{levelCode}")
    Call<Result<Void>> updateBenefit(@Path("levelCode") String levelCode, @Body BenefitForm form);
}
