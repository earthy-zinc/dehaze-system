package com.pei.dehaze.ui.profile.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.AuthAPI;
import com.pei.dehaze.sdk.api.UserAPI;
import com.pei.dehaze.sdk.api.FavoriteAPI;
import com.pei.dehaze.sdk.api.TaskAPI;
import com.pei.dehaze.sdk.api.MemberAPI;
import com.pei.dehaze.sdk.model.favorite.FavoriteCountVO;
import com.pei.dehaze.sdk.model.member.MemberProfileVO;
import com.pei.dehaze.sdk.model.task.TaskQuery;
import com.pei.dehaze.sdk.model.task.TaskVO;
import com.pei.dehaze.sdk.model.user.UserInfo;
import com.pei.dehaze.sdk.utils.TokenManager;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.PageResult;

import java.util.List;

public class ProfileViewModel extends BaseViewModel {

    private final MutableLiveData<UserInfo> userInfo = new MutableLiveData<>();
    private final MutableLiveData<Boolean> logoutSuccess = new MutableLiveData<>(false);
    private final MutableLiveData<Boolean> notLoggedIn = new MutableLiveData<>(false);
    private final MutableLiveData<Long> quotaRemaining = new MutableLiveData<>();
    private final MutableLiveData<Long> favoriteCount = new MutableLiveData<>();
    private final MutableLiveData<Long> taskTotal = new MutableLiveData<>();

    public void loadUserInfo() {
        if (!TokenManager.hasToken()) {
            notLoggedIn.setValue(true);
            loading.setValue(false);
            return;
        }
        notLoggedIn.setValue(false);
        UserAPI.getInfo(RepositoryAdapters.wrap(withLoading(userInfo::postValue)));
    }

    public void loadStats() {
        loadFavoriteCount();
        loadTaskTotal();
        loadMemberProfile();
    }

    private void loadFavoriteCount() {
        FavoriteAPI.getCount(null, RepositoryAdapters.wrap(
                withLoading((OnSuccess<List<FavoriteCountVO>>) data -> {
                    long total = 0;
                    if (data != null) {
                        for (FavoriteCountVO item : data) {
                            total += item.getCount() != null ? item.getCount() : 0L;
                        }
                    }
                    favoriteCount.postValue(total);
                })));
    }

    private void loadTaskTotal() {
        TaskQuery query = new TaskQuery();
        query.setPageNum(1);
        query.setPageSize(1);
        TaskAPI.getTaskPage(query, RepositoryAdapters.wrap(
                withLoading((OnSuccess<PageResult<TaskVO>>) data ->
                        taskTotal.postValue(data != null ? data.getTotal() : 0L))));
    }

    private void loadMemberProfile() {
        MemberAPI.getProfile(RepositoryAdapters.wrap(
                withLoading((OnSuccess<MemberProfileVO>) data -> {
                    if (data != null) {
                        int quota = data.getMonthlyDehazeQuota() != null ? data.getMonthlyDehazeQuota() : 0;
                        int used = data.getMonthlyDehazeUsed() != null ? data.getMonthlyDehazeUsed() : 0;
                        quotaRemaining.postValue((long) Math.max(0, quota - used));
                    }
                })));
    }

    public void logout() {
        AuthAPI.logout(RepositoryAdapters.wrap(withLoading(v -> logoutSuccess.postValue(true))));
    }

    public LiveData<UserInfo> getUserInfo() { return userInfo; }
    public LiveData<Boolean> getLogoutSuccess() { return logoutSuccess; }
    public LiveData<Boolean> getNotLoggedIn() { return notLoggedIn; }
    public LiveData<Long> getQuotaRemaining() { return quotaRemaining; }
    public LiveData<Long> getFavoriteCount() { return favoriteCount; }
    public LiveData<Long> getTaskTotal() { return taskTotal; }
}
