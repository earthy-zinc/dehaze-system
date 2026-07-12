package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.DictRepository;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.dict.DictTypeForm;
import com.pei.dehaze.sdk.model.dict.DictTypePageVO;
import com.pei.dehaze.sdk.model.dict.DictTypeQuery;

import java.util.List;

public class DictTypeViewModel extends ViewModel {

    private final DictRepository dictRepository;

    private final MutableLiveData<List<DictTypePageVO>> dictTypeList = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>();
    private final MutableLiveData<DictTypeForm> dictTypeForm = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> actionResult = new MutableLiveData<>();

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
        loading.setValue(true);
        DictTypeQuery query = new DictTypeQuery();
        query.setPageNum(currentPage);
        query.setPageSize(pageSize);
        query.setKeywords(currentKeywords);
        dictRepository.getDictTypePage(query, new DictRepository.DictTypePageCallback() {
            @Override
            public void onSuccess(PageResult<DictTypePageVO> page) {
                dictTypeList.postValue(page.getList());
                total.postValue(page.getTotal());
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadDictTypeForm(int id) {
        loading.setValue(true);
        dictRepository.getDictTypeForm(id, new DictRepository.DictTypeFormCallback() {
            @Override
            public void onSuccess(DictTypeForm form) {
                dictTypeForm.postValue(form);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void addDictType(DictTypeForm form) {
        loading.setValue(true);
        dictRepository.addDictType(form, new DictRepository.DictTypeActionCallback() {
            @Override
            public void onSuccess() {
                actionResult.postValue("新增字典类型成功");
                loading.postValue(false);
                queryPage();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void updateDictType(int id, DictTypeForm form) {
        loading.setValue(true);
        dictRepository.updateDictType(id, form, new DictRepository.DictTypeActionCallback() {
            @Override
            public void onSuccess() {
                actionResult.postValue("修改字典类型成功");
                loading.postValue(false);
                queryPage();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void deleteDictType(int id) {
        loading.setValue(true);
        dictRepository.deleteDictType(id, new DictRepository.DictTypeActionCallback() {
            @Override
            public void onSuccess() {
                actionResult.postValue("删除字典类型成功");
                loading.postValue(false);
                queryPage();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
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

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }

    public LiveData<String> getActionResult() {
        return actionResult;
    }

    public int getCurrentPage() {
        return currentPage;
    }

    public int getPageSize() {
        return pageSize;
    }
}
