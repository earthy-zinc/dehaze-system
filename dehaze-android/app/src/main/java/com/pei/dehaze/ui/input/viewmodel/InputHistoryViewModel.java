package com.pei.dehaze.ui.input.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.InputHistoryRepository;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.input_history.InputHistoryForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryQuery;
import com.pei.dehaze.sdk.model.input_history.InputHistoryUpdateForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryVO;
import com.pei.dehaze.sdk.model.input_history.InputSource;
import com.pei.dehaze.sdk.model.input_history.SyncResultVO;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * 图像输入历史 ViewModel
 */
public class InputHistoryViewModel extends BaseViewModel {

    private final InputHistoryRepository repository;

    private final MutableLiveData<List<InputHistoryVO>> historyList = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>(0L);
    private final MutableLiveData<SyncResultVO> syncResult = new MutableLiveData<>();
    private final MutableLiveData<FileInfo> uploadedFile = new MutableLiveData<>();

    private int pageNum = 1;
    private final int pageSize = 10;
    private String keywords = "";
    private InputSource inputSource;
    private Boolean favoriteOnly = false;

    public InputHistoryViewModel() {
        repository = new InputHistoryRepository();
    }

    public LiveData<List<InputHistoryVO>> getHistoryList() {
        return historyList;
    }

    public LiveData<Long> getTotal() {
        return total;
    }

    public LiveData<SyncResultVO> getSyncResult() {
        return syncResult;
    }

    public LiveData<FileInfo> getUploadedFile() {
        return uploadedFile;
    }

    public void clearSyncResult() {
        syncResult.setValue(null);
    }

    public void clearUploadedFile() {
        uploadedFile.setValue(null);
    }

    public void loadHistory() {
        InputHistoryQuery query = buildQuery();
        repository.listHistory(query, withLoading(data -> {
            historyList.postValue(data != null ? data.getList() : new ArrayList<>());
            total.postValue(data != null ? data.getTotal() : 0L);
        }));
    }

    private InputHistoryQuery buildQuery() {
        InputHistoryQuery query = new InputHistoryQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setKeywords(keywords);
        query.setInputSource(inputSource);
        query.setFavoriteOnly(favoriteOnly);
        return query;
    }

    public void setQueryParams(String keywords, InputSource inputSource, Boolean favoriteOnly) {
        this.keywords = keywords == null ? "" : keywords.trim();
        this.inputSource = inputSource;
        this.favoriteOnly = favoriteOnly;
        this.pageNum = 1;
    }

    public void resetQuery() {
        this.keywords = "";
        this.inputSource = null;
        this.favoriteOnly = false;
        this.pageNum = 1;
    }

    public void nextPage() {
        long totalVal = total.getValue() != null ? total.getValue() : 0L;
        int totalPages = (int) Math.ceil(totalVal * 1.0 / pageSize);
        if (pageNum < totalPages) {
            pageNum++;
            loadHistory();
        }
    }

    public void prevPage() {
        if (pageNum > 1) {
            pageNum--;
            loadHistory();
        }
    }

    public int getPageNum() {
        return pageNum;
    }

    public int getPageSize() {
        return pageSize;
    }

    public void createHistory(InputHistoryForm form) {
        repository.createHistory(form, withLoading(v -> {
            operationResult.postValue("新增历史记录成功");
            loadHistory();
        }));
    }

    public void updateHistory(long id, InputHistoryUpdateForm form) {
        repository.updateHistory(id, form, withLoading(v -> {
            operationResult.postValue("修改历史记录成功");
            loadHistory();
        }));
    }

    public void deleteHistory(long id) {
        repository.deleteHistory(id, withLoading(v -> {
            operationResult.postValue("删除历史记录成功");
            loadHistory();
        }));
    }

    public void batchDeleteHistory(List<Long> ids) {
        repository.batchDeleteHistory(ids, withLoading(v -> {
            operationResult.postValue("批量删除成功");
            loadHistory();
        }));
    }

    public void clearHistory() {
        repository.clearHistory(withLoading(v -> {
            operationResult.postValue("清空历史记录成功");
            loadHistory();
        }));
    }

    public void syncHistory(List<InputHistoryForm> items) {
        repository.syncHistory(items, withLoading(syncResult::postValue));
    }

    public void uploadFile(File file) {
        repository.uploadFile(file, withLoading(uploadedFile::postValue));
    }
}
