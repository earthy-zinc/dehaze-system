package com.pei.dehaze.ui.common;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.network.CallTracker;
import com.pei.dehaze.sdk.network.RequestScope;

/**
 * ViewModel 基类，提供统一的 loading/error/operationResult 状态管理
 */
public abstract class BaseViewModel extends ViewModel {
    protected final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    protected final MutableLiveData<String> error = new MutableLiveData<>();
    protected final MutableLiveData<String> operationResult = new MutableLiveData<>();

    /** 当前 ViewModel 发起的进行中请求跟踪器，onCleared 时仅取消自身请求。 */
    private final CallTracker callTracker = new CallTracker();

    public LiveData<Boolean> getLoading() { return loading; }
    public LiveData<String> getError() { return error; }
    public LiveData<String> getOperationResult() { return operationResult; }

    public void clearError() { error.setValue(null); }
    public void clearOperationResult() { operationResult.setValue(null); }

    @Override
    protected void onCleared() {
        super.onCleared();
        callTracker.cancelAll();
    }

    /**
     * 成功回调函数式接口。
     * 自定义而非使用 java.util.function.Consumer，以兼容 minSdk 23（Consumer 需 API 24）。
     *
     * @param <T> 回调数据类型
     */
    @FunctionalInterface
    public interface OnSuccess<T> {
        void accept(T data);
    }

    /**
     * 错误回调函数式接口（自定义以兼容 minSdk 23）。
     */
    @FunctionalInterface
    public interface OnError {
        void accept(String errorMessage);
    }

    /**
     * 通用异步执行器，封装 "设置 loading → 成功处理并复位 loading → 失败上报并复位 loading" 的固定样板。
     *
     * <p>典型用法：
     * <pre>{@code
     * repository.getItems(query, withLoading(data -> {
     *     items.postValue(data.getList());
     *     total.postValue(data.getTotal());
     * }));
     * }</pre>
     *
     * @param onSuccess 成功回调（在主线程回调线程中执行，内部可使用 postValue）
     * @param <T>       回调数据类型
     * @return 包装好的 RepositoryCallback
     */
    protected <T> RepositoryCallback<T> withLoading(OnSuccess<T> onSuccess) {
        return withLoading(onSuccess, error::postValue);
    }

    /**
     * 通用异步执行器（带自定义错误处理），允许调用方对错误消息加工后再上报。
     *
     * @param onSuccess 成功回调
     * @param onError   错误回调（接收原始错误消息，调用方可包装后再 postValue）
     * @param <T>       回调数据类型
     * @return 包装好的 RepositoryCallback
     */
    protected <T> RepositoryCallback<T> withLoading(OnSuccess<T> onSuccess, OnError onError) {
        loading.setValue(true);
        // 将当前 VM 的 tracker 放入请求作用域，供 TrackedCall.enqueue 登记本次请求。
        // withLoading 返回的回调会被调用方立即同步传入 Repository 发起 SDK 调用，
        // 因此 setTracker 与 TrackedCall 的读取发生在同一线程的同步调用链中。
        RequestScope.setTracker(callTracker);
        return new RepositoryCallback<T>() {
            @Override
            public void onSuccess(T data) {
                onSuccess.accept(data);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                onError.accept(errorMessage);
                loading.postValue(false);
            }
        };
    }
}

