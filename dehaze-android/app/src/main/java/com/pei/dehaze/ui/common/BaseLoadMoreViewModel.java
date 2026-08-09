package com.pei.dehaze.ui.common;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import java.util.ArrayList;
import java.util.List;

/**
 * 追加式（无限滚动）分页 ViewModel 基类：维护 pageNum/total，第 1 页替换、其余追加。
 * 子类实现 {@link #loadPage()} 发起请求，成功后调用 {@link #onPageLoaded(List, long)}。
 */
public abstract class BaseLoadMoreViewModel<T> extends BaseViewModel {

    protected final MutableLiveData<List<T>> itemList = new MutableLiveData<>(new ArrayList<>());
    protected int pageNum = 1;
    protected final int pageSize;
    protected long total = 0;

    protected BaseLoadMoreViewModel(int pageSize) {
        this.pageSize = pageSize;
    }

    public LiveData<List<T>> getItemList() { return itemList; }
    public int getPageSize() { return pageSize; }

    /** 子类实现：按当前 pageNum 与筛选条件发起请求，成功后调用 onPageLoaded。 */
    protected abstract void loadPage();

    /** 重新加载（回到第 1 页，整列表替换）。 */
    public void reload() {
        pageNum = 1;
        loadPage();
    }

    /** 加载更多（追加下一页），无更多数据时为空操作。 */
    public void loadMore() {
        if (!hasMore()) {
            return;
        }
        pageNum++;
        loadPage();
    }

    /** 子类在 loadPage 成功回调中调用：第 1 页替换、其余追加，并更新 total。 */
    protected void onPageLoaded(List<T> list, long total) {
        this.total = total;
        if (pageNum == 1) {
            itemList.postValue(list != null ? list : new ArrayList<>());
        } else {
            List<T> merged = new ArrayList<>(currentList());
            if (list != null) merged.addAll(list);
            itemList.postValue(merged);
        }
    }

    public boolean hasMore() {
        return currentList().size() < total;
    }

    protected List<T> currentList() {
        List<T> cur = itemList.getValue();
        return cur != null ? cur : new ArrayList<>();
    }
}
