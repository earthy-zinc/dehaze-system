package com.pei.clarity.sdk.service;

import com.pei.clarity.sdk.model.Result;
import com.pei.clarity.sdk.model.message.NotificationSettings;
import com.pei.clarity.sdk.model.message.NotificationSettingsForm;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.PATCH;

/**
 * 通知设置API服务接口
 * 对齐后端路由：/api/v1/notification-settings
 */
public interface NotificationSettingApiService {

    @GET("/api/v1/notification-settings")
    Call<Result<NotificationSettings>> get();

    @PATCH("/api/v1/notification-settings")
    Call<Result<Void>> update(@Body NotificationSettingsForm form);
}
