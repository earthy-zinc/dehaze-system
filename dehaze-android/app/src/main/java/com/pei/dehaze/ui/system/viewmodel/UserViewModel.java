package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.DeptAPI;
import com.pei.dehaze.sdk.api.RoleAPI;
import com.pei.dehaze.sdk.api.UserAPI;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.EnableStatus;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.role.RoleQuery;
import com.pei.dehaze.sdk.model.user.UserForm;
import com.pei.dehaze.sdk.model.user.UserPageVO;
import com.pei.dehaze.sdk.model.user.UserQuery;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class UserViewModel extends BaseViewModel {

    private final MutableLiveData<List<UserPageVO>> userList = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>(0L);
    private final MutableLiveData<UserForm> userForm = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> deptOptions = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> roleOptions = new MutableLiveData<>();

    private int pageNum = 1;
    private int pageSize = 10;
    private String keywords = "";
    private EnableStatus status;
    private Integer deptId;
    private String startTime;
    private String endTime;

    public void loadUsers() {
        UserQuery query = buildQuery();
        UserAPI.getPage(query, RepositoryAdapters.wrap(withLoading(data -> {
            userList.postValue(data.getList());
            total.postValue(data.getTotal());
        })));
    }

    public void loadUserForm(int userId) {
        UserAPI.getFormData(userId, RepositoryAdapters.wrap(withLoading(userForm::postValue)));
    }

    public void addUser(UserForm form) {
        UserAPI.add(form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("新增用户成功");
            loadUsers();
        })));
    }

    public void updateUser(int id, UserForm form) {
        UserAPI.update(id, form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("修改用户成功");
            loadUsers();
        })));
    }

    public void deleteUsers(List<Long> ids) {
        UserAPI.deleteByIds(ids, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("删除用户成功");
            loadUsers();
        })));
    }

    public void updateUserPassword(int id, String password) {
        UserAPI.updatePassword(id, password,
                RepositoryAdapters.wrap(withLoading(v -> operationResult.postValue("重置密码成功"))));
    }

    public void updateUserStatus(long id, EnableStatus status) {
        UserAPI.updateStatus(id, status, RepositoryAdapters.wrap(new RepositoryCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("状态切换成功");
                loadUsers();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        }));
    }

    public void downloadTemplate(String filePath) {
        UserAPI.downloadTemplate(filePath,
                RepositoryAdapters.wrap(withLoading(v -> operationResult.postValue("模板下载成功:" + filePath))));
    }

    public void exportUsers(String filePath) {
        UserQuery query = buildQuery();
        UserAPI.export(query, filePath,
                RepositoryAdapters.wrap(withLoading(v -> operationResult.postValue("导出成功:" + filePath))));
    }

    public void importUsers(int deptId, File file) {
        UserAPI.importUsers(deptId, file, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("导入成功");
            loadUsers();
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

    public void loadRoleOptions() {
        RoleAPI.getOptions(new RoleQuery(), RepositoryAdapters.wrap(new RepositoryCallback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> data) {
                roleOptions.postValue(data);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        }));
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

    public void setQueryParams(String keywords, EnableStatus status, Integer deptId, String startTime, String endTime) {
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

    public LiveData<UserForm> getUserForm() {
        return userForm;
    }

    public LiveData<List<Option>> getDeptOptions() {
        return deptOptions;
    }

    public LiveData<List<Option>> getRoleOptions() {
        return roleOptions;
    }

    public void clearUserForm() {
        userForm.setValue(null);
    }
}
