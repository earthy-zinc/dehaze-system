package com.pei.dehaze.sdk.network;

/**
 * 请求作用域：在调用线程上传递当前 ViewModel 的 {@link CallTracker}。
 *
 * <p>调用链均为同步执行：{@code BaseViewModel.withLoading} 调用 {@link #setTracker} 后返回回调，
 * 该回调立即被 Repository 同步用于发起 SDK 调用，{@code TrackedCall.enqueue} 在同一线程读取并消费。
 * 因此 setTracker 与读取发生在同一线程的同步调用链中，无需跨线程传递。
 */
public final class RequestScope {

    private static final ThreadLocal<CallTracker> CURRENT = new ThreadLocal<>();

    private RequestScope() {
    }

    public static void setTracker(CallTracker tracker) {
        CURRENT.set(tracker);
    }

    public static CallTracker currentTracker() {
        return CURRENT.get();
    }

    /** 消费并清除当前线程上的 tracker，避免泄漏到未经过 withLoading 的请求。 */
    public static void clear() {
        CURRENT.remove();
    }
}
