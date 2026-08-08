package com.pei.dehaze.ui.messages;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.MessageAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.message.MessageQuery;
import com.pei.dehaze.sdk.model.message.MessageVO;
import com.pei.dehaze.sdk.model.message.ReadAllResult;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.ArrayList;
import java.util.List;

/**
 * 消息列表 ViewModel（Fragment scope）。
 *
 * 消息列表数据来源为后端 /api/v1/messages，未读数由全局
 * {@link UnreadMessageViewModel} 维护，本类不持有 unreadCount LiveData。
 *
 * filter 切换会触发重新加载（对齐后端 type 参数）。
 */
public class MessagesViewModel extends BaseViewModel {

    /** filter 文案与后端 type 的映射，null 表示全部。 */
    private static final String[] FILTER_LABELS = {"全部", "站内信", "公告", "业务"};
    private static final String[] FILTER_TYPES = {null, "inbox", "announcement", "business"};

    private final MutableLiveData<List<MessageVO>> messages = new MutableLiveData<>(new ArrayList<>());
    private final MutableLiveData<Integer> currentFilter = new MutableLiveData<>(0);
    private final MutableLiveData<Boolean> refreshing = new MutableLiveData<>(false);

    /** 标记已读成功的消息 id，供 Fragment 乐观刷新本地列表 */
    private final MutableLiveData<Long> markedReadId = new MutableLiveData<>();
    /** 全部已读成功事件，供 Fragment 乐观刷新本地列表 */
    private final MutableLiveData<Boolean> markedAllRead = new MutableLiveData<>();

    public void loadMessages() {
        refreshing.setValue(true);

        Integer filterIdx = currentFilter.getValue();
        String type = (filterIdx != null && filterIdx >= 0 && filterIdx < FILTER_TYPES.length)
                ? FILTER_TYPES[filterIdx] : null;

        MessageQuery query = new MessageQuery();
        query.setPageNum(1);
        query.setPageSize(50);
        query.setType(type);

        MessageAPI.getPage(query, RepositoryAdapters.wrap(new RepositoryCallback<PageResult<MessageVO>>() {
            @Override
            public void onSuccess(PageResult<MessageVO> data) {
                List<MessageVO> list = data != null && data.getList() != null ? data.getList() : new ArrayList<>();
                messages.postValue(list);
                refreshing.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                refreshing.postValue(false);
            }
        }));
    }

    /**
     * 标记单条已读：调后端 {@code PATCH /api/v1/messages/{id}/_read}，成功后
     * 通过 {@link #markedReadId} 通知 Fragment 乐观刷新列表。
     */
    public void markAsRead(long messageId) {
        MessageAPI.markRead(messageId, RepositoryAdapters.wrap(new RepositoryCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                markedReadId.postValue(messageId);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        }));
    }

    /**
     * 全部标记已读：调后端 {@code PATCH /api/v1/messages/_read-all?type=}，
     * type 取当前 filter。成功后通过 {@link #markedAllRead} 通知 Fragment。
     */
    public void markAllRead() {
        Integer filterIdx = currentFilter.getValue();
        String type = (filterIdx != null && filterIdx >= 0 && filterIdx < FILTER_TYPES.length)
                ? FILTER_TYPES[filterIdx] : null;

        MessageAPI.markAllRead(type, RepositoryAdapters.wrap(new RepositoryCallback<ReadAllResult>() {
            @Override
            public void onSuccess(ReadAllResult data) {
                markedAllRead.postValue(true);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        }));
    }

    public void setFilter(int filter) {
        currentFilter.setValue(filter);
        // filter 变化立即触发重新加载
        loadMessages();
    }

    public static String[] getFilterLabels() {
        return FILTER_LABELS;
    }

    public LiveData<List<MessageVO>> getMessages() {
        return messages;
    }

    public LiveData<Integer> getCurrentFilter() {
        return currentFilter;
    }

    public LiveData<Boolean> getRefreshing() {
        return refreshing;
    }

    public LiveData<Long> getMarkedReadId() {
        return markedReadId;
    }

    public LiveData<Boolean> getMarkedAllRead() {
        return markedAllRead;
    }
}
