package com.pei.dehaze.ui.messages;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.MessageAPI;
import com.pei.dehaze.sdk.model.message.UnreadCountVO;

/**
 * 全局未读消息数 ViewModel（Activity scope）。
 *
 * 持有未读消息数 LiveData，供 MainActivity 观察以更新底部导航角标，
 * 同时供 MessagesFragment 在标记已读后触发刷新。
 *
 * 设计理由：未读数是跨 Tab 的全局状态，不应绑定到 MessagesFragment 的
 * view lifecycle，否则切 Tab 时 ViewModel 销毁，角标消失。
 */
public class UnreadMessageViewModel extends ViewModel {

    private final MutableLiveData<Integer> unreadCount = new MutableLiveData<>(0);

    public LiveData<Integer> getUnreadCount() {
        return unreadCount;
    }

    /**
     * 从后端拉取最新未读消息数。失败时静默处理（不影响用户主流程）。
     */
    public void refresh() {
        MessageAPI.getUnreadCount(RepositoryAdapters.wrap(new RepositoryCallback<UnreadCountVO>() {
            @Override
            public void onSuccess(UnreadCountVO data) {
                unreadCount.postValue(data != null && data.getCount() != null ? data.getCount() : 0);
            }

            @Override
            public void onError(String errorMessage) {
                // 静默失败：未读数刷新失败不打扰用户
            }
        }));
    }
}
