package com.pei.dehaze.sdk.network;

import java.io.IOException;

import okhttp3.Request;
import okio.Timeout;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * 包装 retrofit2.Call，在 {@link #enqueue} 时将自身登记到当前线程的 {@link CallTracker}，
 * 使 ViewModel 可在 onCleared 时精确取消自身发起的请求，而非全局 cancelAll。
 *
 * <p>登记后立即消费 {@link RequestScope}，保证未被 withLoading 发起的请求不会被误登记。
 */
class TrackedCall<T> implements Call<T> {

    private final Call<T> delegate;

    TrackedCall(Call<T> delegate) {
        this.delegate = delegate;
    }

    @Override
    public void enqueue(Callback<T> callback) {
        CallTracker tracker = RequestScope.currentTracker();
        if (tracker != null) {
            tracker.register(this);
            RequestScope.clear();
        }
        delegate.enqueue(callback);
    }

    @Override
    public boolean isExecuted() {
        return delegate.isExecuted();
    }

    @Override
    public void cancel() {
        delegate.cancel();
    }

    @Override
    public boolean isCanceled() {
        return delegate.isCanceled();
    }

    @SuppressWarnings("MethodDoesntCallSuperMethod")
    @Override
    public Call<T> clone() {
        return new TrackedCall<>(delegate.clone());
    }

    @Override
    public Request request() {
        return delegate.request();
    }

    @Override
    public Response<T> execute() throws IOException {
        return delegate.execute();
    }

    @Override
    public Timeout timeout() {
        return delegate.timeout();
    }
}
