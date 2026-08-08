package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.ArrayList;
import java.util.List;

/**
 * 管理模块通用 ViewModel 基类，管理列表/分页/搜索/操作结果
 * @param <T> 列表项类型
 */
public abstract class BaseManageViewModel<T> extends BaseViewModel {

    protected final MutableLiveData<List<T>> itemList = new MutableLiveData<>(new ArrayList<>());
    protected final MutableLiveData<Long> total = new MutableLiveData<>(0L);

    protected int pageNum = 1;
    protected final int pageSize = 10;
    protected String keywords = "";

    public abstract void loadData();

    public void setKeywords(String keywords) {
        this.keywords = keywords != null ? keywords : "";
        this.pageNum = 1;
    }

    public void resetQuery() {
        this.keywords = "";
        this.pageNum = 1;
    }

    public void nextPage() {
        long t = total.getValue() != null ? total.getValue() : 0L;
        if (pageNum < Math.max(1, (int) Math.ceil(t * 1.0 / pageSize))) {
            pageNum++;
            loadData();
        }
    }

    public void prevPage() {
        if (pageNum > 1) {
            pageNum--;
            loadData();
        }
    }

    public int getPageNum() { return pageNum; }
    public int getPageSize() { return pageSize; }

    public LiveData<List<T>> getItemList() { return itemList; }
    public LiveData<Long> getTotal() { return total; }
}
