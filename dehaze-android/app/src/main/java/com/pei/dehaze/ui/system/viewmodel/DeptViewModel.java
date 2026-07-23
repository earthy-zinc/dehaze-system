package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.DeptRepository;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.dept.DeptForm;
import com.pei.dehaze.sdk.model.dept.DeptQuery;
import com.pei.dehaze.sdk.model.dept.DeptVO;

import java.util.ArrayList;
import java.util.List;

public class DeptViewModel extends BaseViewModel {

    private final DeptRepository deptRepository;

    private final MutableLiveData<List<DeptVO>> deptList = new MutableLiveData<>();
    private final MutableLiveData<DeptForm> deptForm = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> deptOptions = new MutableLiveData<>();

    private String keywords = "";
    private Integer status;

    public DeptViewModel() {
        deptRepository = new DeptRepository();
    }

    public void loadDepts() {
        DeptQuery query = buildQuery();
        deptRepository.getDepts(query, withLoading(data ->
                deptList.postValue(data != null ? data : new ArrayList<>())));
    }

    public void loadDeptForm(int id) {
        deptRepository.getDeptForm(id, withLoading(deptForm::postValue));
    }

    public void addDept(DeptForm form) {
        deptRepository.addDept(form, withLoading(v -> {
            operationResult.postValue("新增部门成功");
            loadDepts();
        }));
    }

    public void updateDept(int id, DeptForm form) {
        deptRepository.updateDept(id, form, withLoading(v -> {
            operationResult.postValue("修改部门成功");
            loadDepts();
        }));
    }

    public void deleteDepts(List<Long> ids) {
        deptRepository.deleteDepts(ids, withLoading(v -> {
            operationResult.postValue("删除部门成功");
            loadDepts();
        }));
    }

    public void loadDeptOptions() {
        deptRepository.getDeptOptions(new RepositoryCallback<List<Option>>() {
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
