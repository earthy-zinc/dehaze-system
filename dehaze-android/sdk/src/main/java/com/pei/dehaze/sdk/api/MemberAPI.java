package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
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
import com.pei.dehaze.sdk.model.member.MemberQuery;
import com.pei.dehaze.sdk.model.member.MemberStatusForm;

import java.util.List;

import retrofit2.Call;

/**
 * 会员管理API接口封装
 * 对齐后端路由：/api/v1/members
 */
public class MemberAPI {

    private MemberAPI() {
    }

    public static void getProfile(ApiCallback<MemberProfileVO> callback) {
        Call<Result<MemberProfileVO>> call = DehazeSDK.getInstance().getMemberApiService().getProfile();
        call.enqueue(callback);
    }

    public static void getGrowthLogs(GrowthLogQuery query, ApiCallback<PageResult<GrowthLogVO>> callback) {
        Call<Result<PageResult<GrowthLogVO>>> call = DehazeSDK.getInstance().getMemberApiService().getGrowthLogs(
                query.getPageNum(), query.getPageSize(),
                query.getChangeType(), query.getStartTime(), query.getEndTime());
        call.enqueue(callback);
    }

    public static void getPage(MemberQuery query, ApiCallback<PageResult<MemberPageVO>> callback) {
        Call<Result<PageResult<MemberPageVO>>> call = DehazeSDK.getInstance().getMemberApiService().getPage(
                query.getPageNum(), query.getPageSize(),
                query.getKeywords(), query.getLevelCode(), query.getStatus(),
                query.getExpireTimeStart(), query.getExpireTimeEnd(),
                query.getGrowthMin(), query.getGrowthMax());
        call.enqueue(callback);
    }

    public static void getDetail(long userId, ApiCallback<MemberDetailVO> callback) {
        Call<Result<MemberDetailVO>> call = DehazeSDK.getInstance().getMemberApiService().getDetail(userId);
        call.enqueue(callback);
    }

    public static void adjustLevel(long userId, LevelAdjustForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getMemberApiService().adjustLevel(userId, form);
        call.enqueue(callback);
    }

    public static void adjustGrowth(long userId, GrowthAdjustForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getMemberApiService().adjustGrowth(userId, form);
        call.enqueue(callback);
    }

    public static void updateStatus(long userId, MemberStatusForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getMemberApiService().updateStatus(userId, form);
        call.enqueue(callback);
    }

    public static void listBenefits(ApiCallback<List<BenefitVO>> callback) {
        Call<Result<List<BenefitVO>>> call = DehazeSDK.getInstance().getMemberApiService().listBenefits();
        call.enqueue(callback);
    }

    public static void updateBenefit(String levelCode, BenefitForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getMemberApiService().updateBenefit(levelCode, form);
        call.enqueue(callback);
    }
}
