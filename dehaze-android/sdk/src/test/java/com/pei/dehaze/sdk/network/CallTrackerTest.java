package com.pei.dehaze.sdk.network;

import org.junit.After;
import org.junit.Test;

import retrofit2.Call;
import retrofit2.Callback;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;

/**
 * 验证 P0：请求取消按 ViewModel 隔离，不再误伤全局。
 */
public class CallTrackerTest {

    @After
    public void tearDown() {
        RequestScope.clear();
    }

    @Test
    public void cancelAll仅取消自身登记的请求_不影响其它Tracker() {
        // 模拟 Tab 切换：ProfileFragment 销毁不应取消 MessagesTab 的未读轮询
        CallTracker trackerProfile = new CallTracker();
        CallTracker trackerMessages = new CallTracker();

        Call<?> profileCall = mock(Call.class);
        Call<?> messagesCall = mock(Call.class);
        when(profileCall.isCanceled()).thenReturn(false);
        when(messagesCall.isCanceled()).thenReturn(false);

        trackerProfile.register(profileCall);
        trackerMessages.register(messagesCall);

        trackerProfile.cancelAll();

        verify(profileCall).cancel();
        verify(messagesCall, never()).cancel();
    }

    @Test
    public void cancelAll清空登记_再次cancelAll不会重复取消() {
        CallTracker tracker = new CallTracker();
        Call<?> call = mock(Call.class);
        when(call.isCanceled()).thenReturn(false);

        tracker.register(call);
        tracker.cancelAll();
        tracker.cancelAll();

        verify(call, times(1)).cancel();
    }

    @Test
    public void register忽略空与已取消的请求() {
        CallTracker tracker = new CallTracker();
        Call<?> canceled = mock(Call.class);
        when(canceled.isCanceled()).thenReturn(true);

        tracker.register(null);
        tracker.register(canceled);
        tracker.cancelAll();

        verify(canceled, never()).cancel();
    }

    @Test
    public void trackedCallEnqueue登记到当前作用域Tracker并消费作用域() {
        CallTracker tracker = new CallTracker();
        RequestScope.setTracker(tracker);

        @SuppressWarnings("unchecked")
        Call<String> delegate = mock(Call.class);
        when(delegate.isCanceled()).thenReturn(false);
        @SuppressWarnings("unchecked")
        Callback<String> callback = mock(Callback.class);

        TrackedCall<String> tracked = new TrackedCall<>(delegate);
        tracked.enqueue(callback);

        // RequestScope 被消费，请求已委托入队
        assertNull(RequestScope.currentTracker());
        verify(delegate).enqueue(callback);

        // 登记生效：cancelAll 会经 TrackedCall.cancel() 取消 delegate
        tracker.cancelAll();
        verify(delegate).cancel();
    }

    @Test
    public void 无作用域Tracker时Enqueue不登记但仍正常入队() {
        assertNull(RequestScope.currentTracker());

        @SuppressWarnings("unchecked")
        Call<String> delegate = mock(Call.class);
        @SuppressWarnings("unchecked")
        Callback<String> callback = mock(Callback.class);

        new TrackedCall<>(delegate).enqueue(callback);

        verify(delegate).enqueue(callback);
    }
}
