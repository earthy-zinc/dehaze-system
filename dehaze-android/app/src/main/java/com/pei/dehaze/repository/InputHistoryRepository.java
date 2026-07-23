package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.api.InputHistoryAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.input_history.InputHistoryForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryQuery;
import com.pei.dehaze.sdk.model.input_history.InputHistoryUpdateForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryVO;
import com.pei.dehaze.sdk.model.input_history.SyncResultVO;

import java.io.File;
import java.util.List;

public class InputHistoryRepository {

    public void listHistory(InputHistoryQuery query, RepositoryCallback<PageResult<InputHistoryVO>> callback) {
        InputHistoryAPI.listHistory(query, RepositoryAdapters.wrap(callback));
    }

    public void getHistory(long id, RepositoryCallback<InputHistoryVO> callback) {
        InputHistoryAPI.getHistory(id, RepositoryAdapters.wrap(callback));
    }

    public void createHistory(InputHistoryForm form, RepositoryCallback<InputHistoryVO> callback) {
        InputHistoryAPI.createHistory(form, RepositoryAdapters.wrap(callback));
    }

    public void updateHistory(long id, InputHistoryUpdateForm form, RepositoryCallback<InputHistoryVO> callback) {
        InputHistoryAPI.updateHistory(id, form, RepositoryAdapters.wrap(callback));
    }

    public void deleteHistory(long id, RepositoryCallback<Void> callback) {
        InputHistoryAPI.deleteHistory(id, RepositoryAdapters.wrap(callback));
    }

    public void batchDeleteHistory(List<Long> ids, RepositoryCallback<Void> callback) {
        InputHistoryAPI.batchDeleteHistory(ids, RepositoryAdapters.wrap(callback));
    }

    public void clearHistory(RepositoryCallback<Void> callback) {
        InputHistoryAPI.clearHistory(RepositoryAdapters.wrap(callback));
    }

    public void syncHistory(List<InputHistoryForm> items, RepositoryCallback<SyncResultVO> callback) {
        InputHistoryAPI.syncHistory(items, RepositoryAdapters.wrap(callback));
    }

    public void uploadFile(File file, RepositoryCallback<FileInfo> callback) {
        FileAPI.upload(file, RepositoryAdapters.wrap(callback));
    }
}
