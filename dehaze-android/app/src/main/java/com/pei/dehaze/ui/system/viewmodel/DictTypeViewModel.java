package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.DictRepository;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.dict.DictTypeForm;
import com.pei.dehaze.sdk.model.dict.DictTypePageVO;
import com.pei.dehaze.sdk.model.dict.DictTypeQuery;

import java.util.List;

public class DictTypeViewModel extends BaseViewModel {

    private final DictRepository dictRepository;

    private final MutableLiveData<List<DictTypePageVO>> dictTypeList = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>();
    private final MutableLiveData<DictTypeForm> dictTypeForm = new MutableLiveData<>();

    private int currentPage = 1;
    private int pageSize = 10;
    private String currentKeywords;

    public DictTypeViewModel() {
        dictRepository = new DictRepository();
    }

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
        dictRepository.getDictTypePage(query, withLoading(page -> {
            dictTypeList.postValue(page.getList());
            total.postValue(page.getTotal());
        }));
    }

    public void loadDictTypeForm(int id) {
        dictRepository.getDictTypeForm(id, withLoading(dictTypeForm::postValue));
    }

    public void addDictType(DictTypeForm form) {
        dictRepository.addDictType(form, withLoading(v -> {
            operationResult.postValue("新增字典类型成功");
            queryPage();
        }));
    }

    public void updateDictType(int id, DictTypeForm form) {
        dictRepository.updateDictType(id, form, withLoading(v -> {
            operationResult.postValue("修改字典类型成功");
            queryPage();
        }));
    }

    public void deleteDictType(List<Long> ids) {
        dictRepository.deleteDictType(ids, withLoading(v -> {
            operationResult.postValue("删除字典类型成功");
            queryPage();
        }));
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
