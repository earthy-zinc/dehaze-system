package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.DeptAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.dept.DeptForm;
import com.pei.dehaze.sdk.model.dept.DeptQuery;
import com.pei.dehaze.sdk.model.dept.DeptVO;

import java.util.ArrayList;
import java.util.List;

public class DeptViewModel extends BaseViewModel {

    private final MutableLiveData<List<DeptVO>> deptList = new MutableLiveData<>();
    private final MutableLiveData<DeptForm> deptForm = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> deptOptions = new MutableLiveData<>();

    private String keywords = "";
    private Integer status;

    public void loadDepts() {
        DeptQuery query = buildQuery();
        DeptAPI.getList(query, RepositoryAdapters.wrap(withLoading(data ->
                deptList.postValue(data != null ? data : new ArrayList<>()))));
    }

    public void loadDeptForm(int id) {
        DeptAPI.getFormData(id, RepositoryAdapters.wrap(withLoading(deptForm::postValue)));
    }

    public void addDept(DeptForm form) {
        DeptAPI.add(form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("新增部门成功");
            loadDepts();
        })));
    }

    public void updateDept(int id, DeptForm form) {
        DeptAPI.update(id, form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("修改部门成功");
            loadDepts();
        })));
    }

    public void deleteDepts(List<Long> ids) {
        DeptAPI.deleteByIds(ids, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("删除部门成功");
            loadDepts();
        })));
    }

    public void loadDeptOptions() {
        DeptAPI.getOptions(RepositoryAdapters.wrap(new RepositoryCallback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> data) {
                deptOptions.postValue(data);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        }));
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

    public LiveData<DeptForm> getDeptForm() {
        return deptForm;
    }

    public LiveData<List<Option>> getDeptOptions() {
        return deptOptions;
    }

    public void clearDeptForm() {
        deptForm.setValue(null);
    }
}
