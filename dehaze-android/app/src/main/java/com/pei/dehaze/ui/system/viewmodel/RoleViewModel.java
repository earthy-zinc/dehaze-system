package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.RoleRepository;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.menu.MenuVO;
import com.pei.dehaze.sdk.model.role.RoleForm;
import com.pei.dehaze.sdk.model.role.RolePageVO;
import com.pei.dehaze.sdk.model.role.RoleQuery;

import java.util.ArrayList;
import java.util.List;

public class RoleViewModel extends ViewModel {

    private final RoleRepository roleRepository;

    private final MutableLiveData<List<RolePageVO>> roleList = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>(0L);
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>(false);
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> operationResult = new MutableLiveData<>();
    private final MutableLiveData<RoleForm> roleForm = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> roleOptions = new MutableLiveData<>();
    private final MutableLiveData<List<MenuVO>> menuList = new MutableLiveData<>();
    private final MutableLiveData<List<Integer>> roleMenuIds = new MutableLiveData<>();

    private int pageNum = 1;
    private int pageSize = 10;
    private String keywords = "";

    public RoleViewModel() {
        roleRepository = new RoleRepository();
    }

    public void loadRoles() {
        loading.setValue(true);
        RoleQuery query = buildQuery();
        roleRepository.getRoles(query, new RoleRepository.Callback<PageResult<RolePageVO>>() {
            @Override
            public void onSuccess(PageResult<RolePageVO> data) {
                roleList.postValue(data.getList() != null ? data.getList() : new ArrayList<>());
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

    public void loadRoleForm(int id) {
        loading.setValue(true);
        roleRepository.getRoleForm(id, new RoleRepository.Callback<RoleForm>() {
            @Override
            public void onSuccess(RoleForm data) {
                roleForm.postValue(data);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void addRole(RoleForm form) {
        loading.setValue(true);
        roleRepository.addRole(form, new RoleRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("新增角色成功");
                loading.postValue(false);
                loadRoles();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void updateRole(int id, RoleForm form) {
        loading.setValue(true);
        roleRepository.updateRole(id, form, new RoleRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("修改角色成功");
                loading.postValue(false);
                loadRoles();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void deleteRoles(String ids) {
        loading.setValue(true);
        roleRepository.deleteRoles(ids, new RoleRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("删除角色成功");
                loading.postValue(false);
                loadRoles();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void updateRoleStatus(long id, int status) {
        roleRepository.updateRoleStatus(id, status, new RoleRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("状态切换成功");
                loadRoles();
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    public void loadRoleOptions() {
        roleRepository.getRoleOptions(new RoleRepository.Callback<List<Option>>() {
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

    public void loadMenuList() {
        roleRepository.getMenuList(new RoleRepository.Callback<List<MenuVO>>() {
            @Override
            public void onSuccess(List<MenuVO> data) {
                menuList.postValue(data);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    public void loadRoleMenuIds(int roleId) {
        roleRepository.getRoleMenuIds(roleId, new RoleRepository.Callback<List<Integer>>() {
            @Override
            public void onSuccess(List<Integer> data) {
                roleMenuIds.postValue(data);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    public void assignMenus(int roleId, List<Integer> menuIds) {
        loading.setValue(true);
        roleRepository.updateRoleMenus(roleId, menuIds, new RoleRepository.Callback<Void>() {
            @Override
            public void onSuccess(Void data) {
                operationResult.postValue("权限分配成功");
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    private RoleQuery buildQuery() {
        RoleQuery query = new RoleQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setKeywords(keywords);
        return query;
    }

    public void setKeywords(String keywords) {
        this.keywords = keywords == null ? "" : keywords;
        this.pageNum = 1;
    }

    public void resetQuery() {
        this.keywords = "";
        this.pageNum = 1;
    }

    public void nextPage() {
        long totalVal = total.getValue() != null ? total.getValue() : 0L;
        int totalPages = (int) Math.ceil(totalVal * 1.0 / pageSize);
        if (pageNum < totalPages) {
            pageNum++;
            loadRoles();
        }
    }

    public void prevPage() {
        if (pageNum > 1) {
            pageNum--;
            loadRoles();
        }
    }

    public void jumpToPage(int page) {
        long totalVal = total.getValue() != null ? total.getValue() : 0L;
        int totalPages = Math.max(1, (int) Math.ceil(totalVal * 1.0 / pageSize));
        if (page >= 1 && page <= totalPages) {
            pageNum = page;
            loadRoles();
        }
    }

    public int getPageNum() {
        return pageNum;
    }

    public int getPageSize() {
        return pageSize;
    }

    public LiveData<List<RolePageVO>> getRoleList() {
        return roleList;
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

    public LiveData<RoleForm> getRoleForm() {
        return roleForm;
    }

    public LiveData<List<Option>> getRoleOptions() {
        return roleOptions;
    }

    public LiveData<List<MenuVO>> getMenuList() {
        return menuList;
    }

    public LiveData<List<Integer>> getRoleMenuIds() {
        return roleMenuIds;
    }

    public void clearError() {
        error.setValue(null);
    }

    public void clearOperationResult() {
        operationResult.setValue(null);
    }

    public void clearRoleForm() {
        roleForm.setValue(null);
    }

    public void clearRoleMenuIds() {
        roleMenuIds.setValue(null);
    }

    public void clearMenuList() {
        menuList.setValue(null);
    }
}
