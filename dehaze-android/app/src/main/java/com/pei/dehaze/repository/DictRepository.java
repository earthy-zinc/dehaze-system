package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.DictAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.dict.DictForm;
import com.pei.dehaze.sdk.model.dict.DictPageVO;
import com.pei.dehaze.sdk.model.dict.DictQuery;
import com.pei.dehaze.sdk.model.dict.DictTypeForm;
import com.pei.dehaze.sdk.model.dict.DictTypePageVO;
import com.pei.dehaze.sdk.model.dict.DictTypeQuery;

import java.util.List;

public class DictRepository {

    public void getDictTypePage(DictTypeQuery query, RepositoryCallback<PageResult<DictTypePageVO>> callback) {
        DictAPI.getDictTypePage(query, RepositoryAdapters.wrap(callback));
    }

    public void getDictTypeForm(int id, RepositoryCallback<DictTypeForm> callback) {
        DictAPI.getDictTypeForm(id, RepositoryAdapters.wrap(callback));
    }

    public void addDictType(DictTypeForm form, RepositoryCallback<Void> callback) {
        DictAPI.addDictType(form, RepositoryAdapters.wrap(callback));
    }

    public void updateDictType(int id, DictTypeForm form, RepositoryCallback<Void> callback) {
        DictAPI.updateDictType(id, form, RepositoryAdapters.wrap(callback));
    }

    public void deleteDictType(List<Long> ids, RepositoryCallback<Void> callback) {
        DictAPI.deleteDictTypes(ids, RepositoryAdapters.wrap(callback));
    }

    public void getDictPage(DictQuery query, RepositoryCallback<PageResult<DictPageVO>> callback) {
        DictAPI.getDictPage(query, RepositoryAdapters.wrap(callback));
    }

    public void getDictForm(int id, RepositoryCallback<DictForm> callback) {
        DictAPI.getDictFormData(id, RepositoryAdapters.wrap(callback));
    }

    public void addDict(DictForm form, RepositoryCallback<Void> callback) {
        DictAPI.addDict(form, RepositoryAdapters.wrap(callback));
    }

    public void updateDict(int id, DictForm form, RepositoryCallback<Void> callback) {
        DictAPI.updateDict(id, form, RepositoryAdapters.wrap(callback));
    }

    public void deleteDict(List<Long> ids, RepositoryCallback<Void> callback) {
        DictAPI.deleteDictByIds(ids, RepositoryAdapters.wrap(callback));
    }
}
