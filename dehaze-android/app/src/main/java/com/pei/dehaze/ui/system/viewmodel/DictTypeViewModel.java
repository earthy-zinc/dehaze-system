package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.api.DictAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.dict.DictTypeForm;
import com.pei.dehaze.sdk.model.dict.DictTypePageVO;
import com.pei.dehaze.sdk.model.dict.DictTypeQuery;

import java.util.List;

public class DictTypeViewModel extends BaseViewModel {

    private final MutableLiveData<List<DictTypePageVO>> dictTypeList = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>();
    private final MutableLiveData<DictTypeForm> dictTypeForm = new MutableLiveData<>();

    private int currentPage = 1;
    private int pageSize = 10;
    private String currentKeywords;

    public void loadDictTypes(String keywords) {
        currentKeywords = keywords;
        currentPage = 1;
        queryPage();
    }

    public void loadPage(int pageNum) {
        currentPage = pageNum;
        queryPage();
    }

    private void queryPage() {
        DictTypeQuery query = new DictTypeQuery();
        query.setPageNum(currentPage);
        query.setPageSize(pageSize);
        query.setKeywords(currentKeywords);
        DictAPI.getDictTypePage(query, RepositoryAdapters.wrap(withLoading(page -> {
            dictTypeList.postValue(page.getList());
            total.postValue(page.getTotal());
        })));
    }

    public void loadDictTypeForm(int id) {
        DictAPI.getDictTypeForm(id, RepositoryAdapters.wrap(withLoading(dictTypeForm::postValue)));
    }

    public void addDictType(DictTypeForm form) {
        DictAPI.addDictType(form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("新增字典类型成功");
            queryPage();
        })));
    }

    public void updateDictType(int id, DictTypeForm form) {
        DictAPI.updateDictType(id, form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("修改字典类型成功");
            queryPage();
        })));
    }

    public void deleteDictType(List<Long> ids) {
        DictAPI.deleteDictTypes(ids, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("删除字典类型成功");
            queryPage();
        })));
    }

    public LiveData<List<DictTypePageVO>> getDictTypeList() {
        return dictTypeList;
    }

    public LiveData<Long> getTotal() {
        return total;
    }

    public LiveData<DictTypeForm> getDictTypeForm() {
        return dictTypeForm;
    }

    public int getCurrentPage() {
        return currentPage;
    }

    public int getPageSize() {
        return pageSize;
    }
}
