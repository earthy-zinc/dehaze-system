package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.message.AnnouncementForm;
import com.pei.dehaze.sdk.model.message.AnnouncementQuery;
import com.pei.dehaze.sdk.model.message.AnnouncementSendResult;
import com.pei.dehaze.sdk.model.message.AnnouncementVO;

import retrofit2.Call;

/**
 * 公告管理API接口封装
 * 对齐后端路由：/api/v1/announcements
 */
public class AnnouncementAPI {

    private AnnouncementAPI() {
    }

    public static void getPage(AnnouncementQuery query, ApiCallback<PageResult<AnnouncementVO>> callback) {
        Call<Result<PageResult<AnnouncementVO>>> call = DehazeSDK.getInstance().getAnnouncementApiService().getPage(
                query.getPageNum(), query.getPageSize(),
                query.getTitle(), query.getType(), query.getStatus());
        call.enqueue(callback);
    }

    public static void create(AnnouncementForm form, ApiCallback<Object> callback) {
        Call<Result<Object>> call = DehazeSDK.getInstance().getAnnouncementApiService().create(form);
        call.enqueue(callback);
    }

    public static void getDetail(long id, ApiCallback<AnnouncementVO> callback) {
        Call<Result<AnnouncementVO>> call = DehazeSDK.getInstance().getAnnouncementApiService().getDetail(id);
        call.enqueue(callback);
    }

    public static void update(long id, AnnouncementForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getAnnouncementApiService().update(id, form);
        call.enqueue(callback);
    }

    public static void deleteById(long id, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getAnnouncementApiService().deleteById(id);
        call.enqueue(callback);
    }

    public static void send(long id, ApiCallback<AnnouncementSendResult> callback) {
        Call<Result<AnnouncementSendResult>> call = DehazeSDK.getInstance().getAnnouncementApiService().send(id);
        call.enqueue(callback);
    }

    public static void cancel(long id, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getAnnouncementApiService().cancel(id);
        call.enqueue(callback);
    }
}
