package com.pei.clarity.sdk.api;

import com.pei.clarity.sdk.ApiCallback;
import com.pei.clarity.sdk.DehazeSDK;
import com.pei.clarity.sdk.model.Result;
import com.pei.clarity.sdk.model.message.NotificationSettings;
import com.pei.clarity.sdk.model.message.NotificationSettingsForm;

import retrofit2.Call;

/**
 * 通知设置API接口封装
 * 对齐后端路由：/api/v1/notification-settings
 */
public class NotificationSettingAPI {

    private NotificationSettingAPI() {
    }

    public static void get(ApiCallback<NotificationSettings> callback) {
        Call<Result<NotificationSettings>> call = DehazeSDK.getInstance().getNotificationSettingApiService().get();
        call.enqueue(callback);
    }

    public static void update(NotificationSettingsForm form, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getNotificationSettingApiService().update(form);
        call.enqueue(callback);
    }
}
