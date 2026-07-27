package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.api.DictAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.dict.DictForm;
import com.pei.dehaze.sdk.model.dict.DictPageVO;
import com.pei.dehaze.sdk.model.dict.DictQuery;

import java.util.List;

public class DictItemViewModel extends BaseViewModel {

    private final MutableLiveData<List<DictPageVO>> dictList = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>();
    private final MutableLiveData<DictForm> dictForm = new MutableLiveData<>();

    private int currentPage = 1;
    private int pageSize = 10;
    private String currentTypeCode;

    public void loadDicts(String typeCode) {
        currentTypeCode = typeCode;
        currentPage = 1;
        queryPage();
    }

    public void loadPage(int pageNum) {
        currentPage = pageNum;
        queryPage();
    }

    private void queryPage() {
        DictQuery query = new DictQuery();
        query.setPageNum(currentPage);
        query.setPageSize(pageSize);
        query.setTypeCode(currentTypeCode);
        DictAPI.getDictPage(query, RepositoryAdapters.wrap(withLoading(page -> {
            dictList.postValue(page.getList());
            total.postValue(page.getTotal());
        })));
    }

    public void loadDictForm(int id) {
        DictAPI.getDictFormData(id, RepositoryAdapters.wrap(withLoading(dictForm::postValue)));
    }

    public void addDict(DictForm form) {
        DictAPI.addDict(form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("新增字典数据成功");
            queryPage();
        })));
    }

    public void updateDict(int id, DictForm form) {
        DictAPI.updateDict(id, form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("修改字典数据成功");
            queryPage();
        })));
    }

    public void deleteDict(List<Long> ids) {
        DictAPI.deleteDictByIds(ids, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("删除字典数据成功");
            queryPage();
        })));
    }

    public String getCurrentTypeCode() {
        return currentTypeCode;
    }

    public LiveData<List<DictPageVO>> getDictList() {
        return dictList;
    }

    public LiveData<Long> getTotal() {
        return total;
    }

    public LiveData<DictForm> getDictForm() {
        return dictForm;
    }

    public int getCurrentPage() {
        return currentPage;
    }

    public int getPageSize() {
        return pageSize;
    }
}
