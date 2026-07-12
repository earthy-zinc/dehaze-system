package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.api.InputHistoryAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.input_history.InputHistoryForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryQuery;
import com.pei.dehaze.sdk.model.input_history.InputHistoryUpdateForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryVO;
import com.pei.dehaze.sdk.model.input_history.SyncResultVO;
import com.pei.dehaze.sdk.network.ApiException;

import java.io.File;
import java.util.List;

public class InputHistoryRepository {

    public interface Callback<T> {
        void onSuccess(T data);
        void onError(String errorMessage);
    }

    public void listHistory(InputHistoryQuery query, Callback<PageResult<InputHistoryVO>> callback) {
        InputHistoryAPI.listHistory(query, new ApiCallback<PageResult<InputHistoryVO>>() {
            @Override
            public void onSuccess(PageResult<InputHistoryVO> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getHistory(long id, Callback<InputHistoryVO> callback) {
        InputHistoryAPI.getHistory(id, new ApiCallback<InputHistoryVO>() {
            @Override
            public void onSuccess(InputHistoryVO data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void createHistory(InputHistoryForm form, Callback<InputHistoryVO> callback) {
        InputHistoryAPI.createHistory(form, new ApiCallback<InputHistoryVO>() {
            @Override
            public void onSuccess(InputHistoryVO data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void updateHistory(long id, InputHistoryUpdateForm form, Callback<InputHistoryVO> callback) {
        InputHistoryAPI.updateHistory(id, form, new ApiCallback<InputHistoryVO>() {
            @Override
            public void onSuccess(InputHistoryVO data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void deleteHistory(long id, Callback<Void> callback) {
        InputHistoryAPI.deleteHistory(id, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void batchDeleteHistory(List<Long> ids, Callback<Void> callback) {
        InputHistoryAPI.batchDeleteHistory(ids, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void clearHistory(Callback<Void> callback) {
        InputHistoryAPI.clearHistory(new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void syncHistory(List<InputHistoryForm> items, Callback<SyncResultVO> callback) {
        InputHistoryAPI.syncHistory(items, new ApiCallback<SyncResultVO>() {
            @Override
            public void onSuccess(SyncResultVO data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void uploadFile(File file, Callback<FileInfo> callback) {
        FileAPI.upload(file, new ApiCallback<FileInfo>() {
            @Override
            public void onSuccess(FileInfo data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }
}
