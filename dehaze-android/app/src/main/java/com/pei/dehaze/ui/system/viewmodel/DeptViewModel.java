package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.sdk.model.dept.DeptVO;
import com.pei.dehaze.repository.DeptRepository;

import java.util.List;

public class DeptViewModel extends ViewModel {
    
    private final DeptRepository deptRepository;
    
    private final MutableLiveData<List<DeptVO>> deptList = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> error = new MutableLiveData<>();
    
    public DeptViewModel() {
        deptRepository = new DeptRepository();
    }
    
    public void loadDepts() {
        loading.setValue(true);
        deptRepository.getDepts(new DeptRepository.DeptCallback() {
            @Override
            public void onSuccess(List<DeptVO> depts) {
                deptList.postValue(depts);
                loading.postValue(false);
            }
            
            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
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
}