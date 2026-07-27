package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.DatasetAPI;
import com.pei.dehaze.sdk.model.dataset.BatchDeleteForm;

import java.util.List;

public class DatasetRepository {

    public void batchDeleteDatasets(List<Long> ids, RepositoryCallback<Void> callback) {
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(ids);
        DatasetAPI.batchDelete(form, RepositoryAdapters.wrap(callback));
    }

    public void batchDeleteItems(List<Long> ids, RepositoryCallback<Void> callback) {
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(ids);
        DatasetAPI.batchDeleteItems(form, RepositoryAdapters.wrap(callback));
    }

    public void batchDeleteItemFiles(List<Long> ids, RepositoryCallback<Void> callback) {
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(ids);
        DatasetAPI.batchDeleteItemFiles(form, RepositoryAdapters.wrap(callback));
    }
}
