package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.DictRepository;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.dict.DictForm;
import com.pei.dehaze.sdk.model.dict.DictPageVO;
import com.pei.dehaze.sdk.model.dict.DictQuery;

import java.util.List;

public class DictItemViewModel extends ViewModel {

    private final DictRepository dictRepository;

    private final MutableLiveData<List<DictPageVO>> dictList = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>();
    private final MutableLiveData<DictForm> dictForm = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> actionResult = new MutableLiveData<>();

    private int currentPage = 1;
    private int pageSize = 10;
    private String currentTypeCode;

    public DictItemViewModel() {
        dictRepository = new DictRepository();
    }

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
        loading.setValue(true);
        DictQuery query = new DictQuery();
        query.setPageNum(currentPage);
        query.setPageSize(pageSize);
        query.setTypeCode(currentTypeCode);
        dictRepository.getDictPage(query, new DictRepository.DictPageCallback() {
            @Override
            public void onSuccess(PageResult<DictPageVO> page) {
                dictList.postValue(page.getList());
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

    public void loadDictForm(int id) {
        loading.setValue(true);
        dictRepository.getDictForm(id, new DictRepository.DictFormCallback() {
            @Override
            public void onSuccess(DictForm form) {
                dictForm.postValue(form);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void addDict(DictForm form) {
        loading.setValue(true);
        dictRepository.addDict(form, new DictRepository.DictActionCallback() {
            @Override
            public void onSuccess() {
                actionResult.postValue("新增字典数据成功");
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

    public void updateDict(int id, DictForm form) {
        loading.setValue(true);
        dictRepository.updateDict(id, form, new DictRepository.DictActionCallback() {
            @Override
            public void onSuccess() {
                actionResult.postValue("修改字典数据成功");
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

    public void deleteDict(int id) {
        loading.setValue(true);
        dictRepository.deleteDict(id, new DictRepository.DictActionCallback() {
            @Override
            public void onSuccess() {
                actionResult.postValue("删除字典数据成功");
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
