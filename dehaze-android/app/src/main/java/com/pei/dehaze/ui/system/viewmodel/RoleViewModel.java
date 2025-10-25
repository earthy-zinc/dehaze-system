package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.sdk.model.role.RolePageVO;
import com.pei.dehaze.repository.RoleRepository;

import java.util.List;

public class RoleViewModel extends ViewModel {
    
    private final RoleRepository roleRepository;
    
    private final MutableLiveData<List<RolePageVO>> roleList = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> error = new MutableLiveData<>();
    
    public RoleViewModel() {
        roleRepository = new RoleRepository();
    }
    
    public void loadRoles() {
        loading.setValue(true);
        roleRepository.getRoles(new RoleRepository.RoleCallback() {
            @Override
            public void onSuccess(List<RolePageVO> roles) {
                roleList.postValue(roles);
                loading.postValue(false);
            }
            
            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }
    
    public LiveData<List<RolePageVO>> getRoleList() {
        return roleList;
    }
    
    public LiveData<Boolean> getLoading() {
        return loading;
    }
    
    public LiveData<String> getError() {
        return error;
    }
}