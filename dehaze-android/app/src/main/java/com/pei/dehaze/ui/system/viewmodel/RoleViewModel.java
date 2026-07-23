package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RoleRepository;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.model.EnableStatus;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.menu.MenuVO;
import com.pei.dehaze.sdk.model.role.RoleForm;
import com.pei.dehaze.sdk.model.role.RolePageVO;
import com.pei.dehaze.sdk.model.role.RoleQuery;

import java.util.ArrayList;
import java.util.List;

public class RoleViewModel extends BaseViewModel {

    private final RoleRepository roleRepository;

    private final MutableLiveData<List<RolePageVO>> roleList = new MutableLiveData<>();
    private final MutableLiveData<Long> total = new MutableLiveData<>(0L);
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
        RoleQuery query = buildQuery();
        roleRepository.getRoles(query, withLoading(data -> {
            roleList.postValue(data.getList());
            total.postValue(data.getTotal());
        }));
    }

    public void loadRoleForm(int id) {
        roleRepository.getRoleForm(id, withLoading(roleForm::postValue));
    }

    public void addRole(RoleForm form) {
        roleRepository.addRole(form, withLoading(v -> {
            operationResult.postValue("新增角色成功");
            loadRoles();
        }));
    }

    public void updateRole(int id, RoleForm form) {
        roleRepository.updateRole(id, form, withLoading(v -> {
            operationResult.postValue("修改角色成功");
            loadRoles();
        }));
    }

    public void deleteRoles(List<Long> ids) {
        roleRepository.deleteRoles(ids, withLoading(v -> {
            operationResult.postValue("删除角色成功");
            loadRoles();
        }));
    }

    public void updateRoleStatus(long id, EnableStatus status) {
        roleRepository.updateRoleStatus(id, status, new RepositoryCallback<Void>() {
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
        roleRepository.getRoleOptions(new RepositoryCallback<List<Option>>() {
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
        roleRepository.getMenuList(new RepositoryCallback<List<MenuVO>>() {
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
        roleRepository.getRoleMenuIds(roleId, new RepositoryCallback<List<Integer>>() {
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
        roleRepository.updateRoleMenus(roleId, menuIds,
                withLoading(v -> operationResult.postValue("权限分配成功")));
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
