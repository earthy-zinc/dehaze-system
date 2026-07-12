package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.DeptRepository;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.dept.DeptForm;
import com.pei.dehaze.sdk.model.dept.DeptQuery;
import com.pei.dehaze.sdk.model.dept.DeptVO;

import java.util.ArrayList;
import java.util.List;

public class DeptViewModel extends ViewModel {

    private final DeptRepository deptRepository;

    private final MutableLiveData<List<DeptVO>> deptList = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> operationResult = new MutableLiveData<>();
    private final MutableLiveData<DeptForm> deptForm = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> deptOptions = new MutableLiveData<>();

    private String keywords = "";
    private Integer status;

    public DeptViewModel() {
        deptRepository = new DeptRepository();
    }

    public void loadDepts() {
        loading.setValue(true);
        DeptQuery query = buildQuery();
        deptRepository.getDepts(query, new DeptRepository.Callback<List<DeptVO>>() {
            @Override
            public void onSuccess(List<DeptVO> data) {
                deptList.postValue(data != null ? data : new ArrayList<>());
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadDeptForm(int id) {
        loading.setValue(true);
        deptRepository.getDeptForm(id, new DeptRepository.Callback<DeptForm>() {
            @Override
            public void onSuccess(DeptForm data) {
                deptForm.postValue(data);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void addDept(DeptForm form) {
        loading.setValue(true);
        deptRepository.addDept(form, new DeptRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("新增部门成功");
                loading.postValue(false);
                loadDepts();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void updateDept(int id, DeptForm form) {
        loading.setValue(true);
        deptRepository.updateDept(id, form, new DeptRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("修改部门成功");
                loading.postValue(false);
                loadDepts();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void deleteDepts(String ids) {
        loading.setValue(true);
        deptRepository.deleteDepts(ids, new DeptRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("删除部门成功");
                loading.postValue(false);
                loadDepts();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadDeptOptions() {
        deptRepository.getDeptOptions(new DeptRepository.Callback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> data) {
                deptOptions.postValue(data);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    private DeptQuery buildQuery() {
        DeptQuery query = new DeptQuery();
        query.setKeywords(keywords);
        query.setStatus(status);
        return query;
    }

    public void setQueryParams(String keywords, Integer status) {
        this.keywords = keywords == null ? "" : keywords;
        this.status = status;
    }

    public void resetQuery() {
        this.keywords = "";
        this.status = null;
    }

    public LiveData<List<DeptVO>> getDeptList() {
        return deptList;
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

    public LiveData<DeptForm> getDeptForm() {
        return deptForm;
    }

    public LiveData<List<Option>> getDeptOptions() {
        return deptOptions;
    }

    public void clearError() {
        error.setValue(null);
    }

    public void clearOperationResult() {
        operationResult.setValue(null);
    }

    public void clearDeptForm() {
        deptForm.setValue(null);
    }
}
