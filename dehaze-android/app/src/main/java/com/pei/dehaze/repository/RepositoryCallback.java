package com.pei.dehaze.repository;

/**
 * Repository 层统一回调接口
 * <p>
 * 所有 Repository 方法均使用此接口作为回调，替代各 Repository 内部重复定义的 Callback/XxxCallback 接口。
 * onSuccess 接收强类型数据；onError 接收统一格式的错误消息（"[code] message" 或网络错误描述）。
 *
 * @param <T> 业务数据类型
 */
public interface RepositoryCallback<T> {
    /**
     * 业务成功时调用
     *
     * @param data 业务数据，可能为 null（如 Void 返回类型）
     */
    void onSuccess(T data);

    /**
     * 业务失败或网络错误时调用
     *
     * @param errorMessage 统一格式的错误消息
     */
    void onError(String errorMessage);
}
