package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.UserRepository;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.user.UserForm;
import com.pei.dehaze.sdk.model.user.UserPageVO;
import com.pei.dehaze.sdk.model.user.UserQuery;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class UserViewModel extends ViewModel {

    private final UserRepository userRepository;

    private final MutableLiveData<List<UserPageVO>> userList = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>(0L);
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> operationResult = new MutableLiveData<>();
    private final MutableLiveData<UserForm> userForm = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> deptOptions = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> roleOptions = new MutableLiveData<>();

    private int pageNum = 1;
    private int pageSize = 10;
    private String keywords = "";
    private Integer status;
    private Integer deptId;
    private String startTime;
    private String endTime;

    public UserViewModel() {
        userRepository = new UserRepository();
    }

    public void loadUsers() {
        loading.setValue(true);
        UserQuery query = buildQuery();
        userRepository.getUsers(query, new UserRepository.Callback<PageResult<UserPageVO>>() {
            @Override
            public void onSuccess(PageResult<UserPageVO> data) {
                userList.postValue(data.getList() != null ? data.getList() : new ArrayList<>());
                total.postValue(data.getTotal());
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadUserForm(int userId) {
        loading.setValue(true);
        userRepository.getUserForm(userId, new UserRepository.Callback<UserForm>() {
            @Override
            public void onSuccess(UserForm data) {
                userForm.postValue(data);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void addUser(UserForm form) {
        loading.setValue(true);
        userRepository.addUser(form, new UserRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("新增用户成功");
                loading.postValue(false);
                loadUsers();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void updateUser(int id, UserForm form) {
        loading.setValue(true);
        userRepository.updateUser(id, form, new UserRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("修改用户成功");
                loading.postValue(false);
                loadUsers();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void deleteUsers(String ids) {
        loading.setValue(true);
        userRepository.deleteUsers(ids, new UserRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("删除用户成功");
                loading.postValue(false);
                loadUsers();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void updateUserPassword(int id, String password) {
        loading.setValue(true);
        userRepository.updateUserPassword(id, password, new UserRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("重置密码成功");
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void updateUserStatus(long id, int status) {
        userRepository.updateUserStatus(id, status, new UserRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("状态切换成功");
                loadUsers();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    public void downloadTemplate(String filePath) {
        loading.setValue(true);
        userRepository.downloadTemplate(filePath, new UserRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("模板下载成功:" + filePath);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void exportUsers(String filePath) {
        loading.setValue(true);
        UserQuery query = buildQuery();
        userRepository.exportUsers(query, filePath, new UserRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("导出成功:" + filePath);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void importUsers(int deptId, File file) {
        loading.setValue(true);
        userRepository.importUsers(deptId, file, new UserRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("导入成功");
                loading.postValue(false);
                loadUsers();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadDeptOptions() {
        userRepository.getDeptOptions(new UserRepository.Callback<List<Option>>() {
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

    public void loadRoleOptions() {
        userRepository.getRoleOptions(new UserRepository.Callback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> data) {
                roleOptions.postValue(data);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    private UserQuery buildQuery() {
        UserQuery query = new UserQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setKeywords(keywords);
        query.setStatus(status);
        query.setDeptId(deptId);
        query.setStartTime(startTime);
        query.setEndTime(endTime);
        return query;
    }

    public void setQueryParams(String keywords, Integer status, Integer deptId, String startTime, String endTime) {
        this.keywords = keywords == null ? "" : keywords;
        this.status = status;
        this.deptId = deptId;
        this.startTime = startTime;
        this.endTime = endTime;
        this.pageNum = 1;
    }

    public void resetQuery() {
        this.keywords = "";
        this.status = null;
        this.deptId = null;
        this.startTime = null;
        this.endTime = null;
        this.pageNum = 1;
    }

    public void nextPage() {
        long totalVal = total.getValue() != null ? total.getValue() : 0L;
        int totalPages = (int) Math.ceil(totalVal * 1.0 / pageSize);
        if (pageNum < totalPages) {
            pageNum++;
            loadUsers();
        }
    }

    public void prevPage() {
        if (pageNum > 1) {
            pageNum--;
            loadUsers();
        }
    }

    public void jumpToPage(int page) {
        long totalVal = total.getValue() != null ? total.getValue() : 0L;
        int totalPages = Math.max(1, (int) Math.ceil(totalVal * 1.0 / pageSize));
        if (page >= 1 && page <= totalPages) {
            pageNum = page;
            loadUsers();
        }
    }

    public int getPageNum() {
        return pageNum;
    }

    public int getPageSize() {
        return pageSize;
    }

    public void setPageSize(int size) {
        this.pageSize = size;
        this.pageNum = 1;
    }

    public LiveData<List<UserPageVO>> getUserList() {
        return userList;
    }

    public LiveData<Long> getTotal() {
        return total;
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

    public LiveData<UserForm> getUserForm() {
        return userForm;
    }

    public LiveData<List<Option>> getDeptOptions() {
        return deptOptions;
    }

    public LiveData<List<Option>> getRoleOptions() {
        return roleOptions;
    }

    public void clearError() {
        error.setValue(null);
    }

    public void clearOperationResult() {
        operationResult.setValue(null);
    }

    public void clearUserForm() {
        userForm.setValue(null);
    }
}
