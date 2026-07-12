package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.DeptAPI;
import com.pei.dehaze.sdk.api.RoleAPI;
import com.pei.dehaze.sdk.api.UserAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.user.UserForm;
import com.pei.dehaze.sdk.model.user.UserPageVO;
import com.pei.dehaze.sdk.model.user.UserQuery;
import com.pei.dehaze.sdk.model.role.RoleQuery;
import com.pei.dehaze.sdk.network.ApiException;

import java.io.File;
import java.util.List;

public class UserRepository {

    public interface Callback<T> {
        void onSuccess(T data);
        void onError(String errorMessage);
    }

    public void getUsers(UserQuery query, Callback<PageResult<UserPageVO>> callback) {
        UserAPI.getPage(query, new ApiCallback<PageResult<UserPageVO>>() {
            @Override
            public void onSuccess(PageResult<UserPageVO> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getUserForm(int userId, Callback<UserForm> callback) {
        UserAPI.getFormData(userId, new ApiCallback<UserForm>() {
            @Override
            public void onSuccess(UserForm data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void addUser(UserForm form, Callback<Void> callback) {
        UserAPI.add(form, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void updateUser(int id, UserForm form, Callback<Void> callback) {
        UserAPI.update(id, form, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void deleteUsers(String ids, Callback<Void> callback) {
        UserAPI.deleteByIds(ids, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void updateUserPassword(int id, String password, Callback<Void> callback) {
        UserAPI.updatePassword(id, password, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void updateUserStatus(long id, int status, Callback<Void> callback) {
        UserAPI.updateStatus(id, status, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void downloadTemplate(String filePath, Callback<Void> callback) {
        UserAPI.downloadTemplate(filePath, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void exportUsers(UserQuery query, String filePath, Callback<Void> callback) {
        UserAPI.export(query, filePath, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void importUsers(int deptId, File file, Callback<Void> callback) {
        UserAPI.importUsers(deptId, file, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getDeptOptions(Callback<List<Option>> callback) {
        DeptAPI.getOptions(new ApiCallback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getRoleOptions(Callback<List<Option>> callback) {
        RoleQuery query = new RoleQuery();
        RoleAPI.getOptions(query, new ApiCallback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }
}
