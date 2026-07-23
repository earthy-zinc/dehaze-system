package com.pei.dehaze.repository;

import androidx.annotation.NonNull;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.network.ApiException;
import com.pei.dehaze.sdk.utils.ErrorUtils;

import java.util.List;

/**
 * Repository 层回调适配器工具类
 * <p>
 * 将 SDK 的 {@link ApiCallback}（三方法：onSuccess/onError/onFailure）
 * 统一适配为 {@link RepositoryCallback}（两方法：onSuccess/onError），
 * 并使用 {@link ErrorUtils} 生成友好错误消息，消除各 Repository 重复的样板代码。
 */
public final class RepositoryAdapters {

    private RepositoryAdapters() {
    }

    /**
     * 创建一个 ApiCallback 适配器，将结果转发给 RepositoryCallback。
     * 错误消息统一使用 ErrorUtils 解析，格式友好且一致。
     *
     * @param callback Repository 回调
     * @param <T>      业务数据类型
     * @return ApiCallback 实例，传给 SDK 的 XxxAPI 方法
     */
    public static <T> ApiCallback<T> wrap(@NonNull RepositoryCallback<T> callback) {
        return new ApiCallback<T>() {
            @Override
            public void onSuccess(T data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError(ErrorUtils.parseError(new ApiException(0, code, message)));
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(ErrorUtils.parseError(e));
            }
        };
    }

    /**
     * 创建一个 ApiCallback 适配器，自动从 PageResult 中拆出 List 数据。
     * 用于 "分页接口仅消费列表" 的场景，避免各 Repository 重复编写拆包样板。
     *
     * @param callback Repository 回调（接收拆包后的 List）
     * @param <T>      列表元素类型
     * @return ApiCallback 实例，传给 SDK 的 XxxAPI 方法
     */
    public static <T> ApiCallback<PageResult<T>> wrapPage(@NonNull RepositoryCallback<List<T>> callback) {
        return new ApiCallback<PageResult<T>>() {
            @Override
            public void onSuccess(PageResult<T> data) {
                callback.onSuccess(data != null ? data.getList() : null);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError(ErrorUtils.parseError(new ApiException(0, code, message)));
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(ErrorUtils.parseError(e));
            }
        };
    }
}
