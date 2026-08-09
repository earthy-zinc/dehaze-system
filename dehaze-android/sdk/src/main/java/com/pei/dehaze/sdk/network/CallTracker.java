package com.pei.dehaze.sdk.network;

import java.util.concurrent.CopyOnWriteArrayList;

import retrofit2.Call;

/**
 * 跟踪同一发起方（如 ViewModel）发起的进行中网络请求，支持统一取消。
 *
 * <p>用于替代 {@code OkHttpClient.dispatcher().cancelAll()} 这一误伤全局的做法：
 * 每个 ViewModel 持有独立的 CallTracker，{@code onCleared} 时仅取消自身登记的请求，
 * 不会影响其他 Tab 的轮询、后台去雾处理等请求。
 */
public final class CallTracker {

    private final CopyOnWriteArrayList<Call<?>> calls = new CopyOnWriteArrayList<>();

    /** 登记一个进行中的请求，便于后续取消。已取消的请求会被忽略。 */
    public void register(Call<?> call) {
        if (call != null && !call.isCanceled()) {
            calls.add(call);
        }
    }

    /** 取消所有尚未完成的登记请求，并清空登记列表。 */
    public void cancelAll() {
        for (Call<?> call : calls) {
            if (!call.isCanceled()) {
                call.cancel();
            }
        }
        calls.clear();
    }
}
