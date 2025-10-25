package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.sdk.model.user.UserPageVO;
import com.pei.dehaze.repository.UserRepository;

import java.util.List;

public class UserViewModel extends ViewModel {
    
    private final UserRepository userRepository;
    
    private final MutableLiveData<List<UserPageVO>> userList = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> error = new MutableLiveData<>();
    
    public UserViewModel() {
        userRepository = new UserRepository();
    }
    
    public void loadUsers() {
        loading.setValue(true);
        userRepository.getUsers(new UserRepository.UserCallback() {
            @Override
            public void onSuccess(List<UserPageVO> users) {
                userList.postValue(users);
                loading.postValue(false);
            }
            
            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }
    
    public LiveData<List<UserPageVO>> getUserList() {
        return userList;
    }
    
    public LiveData<Boolean> getLoading() {
        return loading;
    }
    
    public LiveData<String> getError() {
        return error;
    }
}