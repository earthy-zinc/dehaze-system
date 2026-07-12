package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.MenuAPI;
import com.pei.dehaze.sdk.api.RoleAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.menu.MenuQuery;
import com.pei.dehaze.sdk.model.menu.MenuVO;
import com.pei.dehaze.sdk.model.role.RoleForm;
import com.pei.dehaze.sdk.model.role.RolePageVO;
import com.pei.dehaze.sdk.model.role.RoleQuery;
import com.pei.dehaze.sdk.network.ApiException;

import java.util.List;

public class RoleRepository {

    public interface Callback<T> {
        void onSuccess(T data);
        void onError(String errorMessage);
    }

    public void getRoles(RoleQuery query, Callback<PageResult<RolePageVO>> callback) {
        RoleAPI.getPage(query, new ApiCallback<PageResult<RolePageVO>>() {
            @Override
            public void onSuccess(PageResult<RolePageVO> data) {
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

    public void getRoleForm(int id, Callback<RoleForm> callback) {
        RoleAPI.getFormData(id, new ApiCallback<RoleForm>() {
            @Override
            public void onSuccess(RoleForm data) {
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

    public void addRole(RoleForm form, Callback<Void> callback) {
        RoleAPI.add(form, new ApiCallback<Void>() {
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

    public void updateRole(int id, RoleForm form, Callback<Void> callback) {
        RoleAPI.update(id, form, new ApiCallback<Void>() {
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

    public void deleteRoles(String ids, Callback<Void> callback) {
        RoleAPI.deleteByIds(ids, new ApiCallback<Void>() {
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

    public void updateRoleStatus(long id, int status, Callback<Void> callback) {
        RoleAPI.updateStatus(id, status, new ApiCallback<Void>() {
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

    public void getRoleMenuIds(int roleId, Callback<List<Integer>> callback) {
        RoleAPI.getRoleMenuIds(roleId, new ApiCallback<List<Integer>>() {
            @Override
            public void onSuccess(List<Integer> data) {
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

    public void updateRoleMenus(int roleId, List<Integer> menuIds, Callback<Void> callback) {
        RoleAPI.updateRoleMenus(roleId, menuIds, new ApiCallback<Void>() {
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

    public void getMenuList(Callback<List<MenuVO>> callback) {
        MenuQuery query = new MenuQuery();
        MenuAPI.getList(query, new ApiCallback<List<MenuVO>>() {
            @Override
            public void onSuccess(List<MenuVO> data) {
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
