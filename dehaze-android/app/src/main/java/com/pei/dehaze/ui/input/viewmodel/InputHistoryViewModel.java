package com.pei.dehaze.ui.input.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.InputHistoryRepository;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.input_history.InputHistoryForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryQuery;
import com.pei.dehaze.sdk.model.input_history.InputHistoryUpdateForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryVO;
import com.pei.dehaze.sdk.model.input_history.SyncResultVO;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * 图像输入历史 ViewModel
 */
public class InputHistoryViewModel extends ViewModel {

    private final InputHistoryRepository repository;

    private final MutableLiveData<List<InputHistoryVO>> historyList = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>(0L);
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> operationResult = new MutableLiveData<>();
    private final MutableLiveData<SyncResultVO> syncResult = new MutableLiveData<>();
    private final MutableLiveData<FileInfo> uploadedFile = new MutableLiveData<>();

    private int pageNum = 1;
    private final int pageSize = 10;
    private String keywords = "";
    private String inputSource;
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

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }

    public LiveData<String> getOperationResult() {
        return operationResult;
    }

    public LiveData<SyncResultVO> getSyncResult() {
        return syncResult;
    }

    public LiveData<FileInfo> getUploadedFile() {
        return uploadedFile;
    }

    public void clearError() {
        error.setValue(null);
    }

    public void clearOperationResult() {
        operationResult.setValue(null);
    }

    public void clearSyncResult() {
        syncResult.setValue(null);
    }

    public void clearUploadedFile() {
        uploadedFile.setValue(null);
    }

    public void loadHistory() {
        loading.setValue(true);
        InputHistoryQuery query = buildQuery();
        repository.listHistory(query, new InputHistoryRepository.Callback<PageResult<InputHistoryVO>>() {
            @Override
            public void onSuccess(PageResult<InputHistoryVO> data) {
                historyList.postValue(data != null && data.getList() != null ? data.getList() : new ArrayList<>());
                total.postValue(data != null ? data.getTotal() : 0L);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
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

    public void setQueryParams(String keywords, String inputSource, Boolean favoriteOnly) {
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
        loading.setValue(true);
        repository.createHistory(form, new InputHistoryRepository.Callback<InputHistoryVO>() {
            @Override
            public void onSuccess(InputHistoryVO data) {
                operationResult.postValue("新增历史记录成功");
                loading.postValue(false);
                loadHistory();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void updateHistory(long id, InputHistoryUpdateForm form) {
        loading.setValue(true);
        repository.updateHistory(id, form, new InputHistoryRepository.Callback<InputHistoryVO>() {
            @Override
            public void onSuccess(InputHistoryVO data) {
                operationResult.postValue("修改历史记录成功");
                loading.postValue(false);
                loadHistory();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void deleteHistory(long id) {
        loading.setValue(true);
        repository.deleteHistory(id, new InputHistoryRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("删除历史记录成功");
                loading.postValue(false);
                loadHistory();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void batchDeleteHistory(List<Long> ids) {
        loading.setValue(true);
        repository.batchDeleteHistory(ids, new InputHistoryRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("批量删除成功");
                loading.postValue(false);
                loadHistory();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void clearHistory() {
        loading.setValue(true);
        repository.clearHistory(new InputHistoryRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("清空历史记录成功");
                loading.postValue(false);
                loadHistory();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void syncHistory(List<InputHistoryForm> items) {
        loading.setValue(true);
        repository.syncHistory(items, new InputHistoryRepository.Callback<SyncResultVO>() {
            @Override
            public void onSuccess(SyncResultVO data) {
                syncResult.postValue(data);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void uploadFile(File file) {
        loading.setValue(true);
        repository.uploadFile(file, new InputHistoryRepository.Callback<FileInfo>() {
            @Override
            public void onSuccess(FileInfo data) {
                uploadedFile.postValue(data);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }
}
