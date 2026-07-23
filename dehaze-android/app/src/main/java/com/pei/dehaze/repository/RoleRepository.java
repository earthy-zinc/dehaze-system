package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.MenuAPI;
import com.pei.dehaze.sdk.api.RoleAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.EnableStatus;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.menu.MenuQuery;
import com.pei.dehaze.sdk.model.menu.MenuVO;
import com.pei.dehaze.sdk.model.role.RoleForm;
import com.pei.dehaze.sdk.model.role.RolePageVO;
import com.pei.dehaze.sdk.model.role.RoleQuery;

import java.util.List;

public class RoleRepository {

    public void getRoles(RoleQuery query, RepositoryCallback<PageResult<RolePageVO>> callback) {
        RoleAPI.getPage(query, RepositoryAdapters.wrap(callback));
    }

    public void getRoleOptions(RepositoryCallback<List<Option>> callback) {
        RoleQuery query = new RoleQuery();
        RoleAPI.getOptions(query, RepositoryAdapters.wrap(callback));
    }

    public void getRoleForm(int id, RepositoryCallback<RoleForm> callback) {
        RoleAPI.getFormData(id, RepositoryAdapters.wrap(callback));
    }

    public void addRole(RoleForm form, RepositoryCallback<Void> callback) {
        RoleAPI.add(form, RepositoryAdapters.wrap(callback));
    }

    public void updateRole(int id, RoleForm form, RepositoryCallback<Void> callback) {
        RoleAPI.update(id, form, RepositoryAdapters.wrap(callback));
    }

    public void deleteRoles(List<Long> ids, RepositoryCallback<Void> callback) {
        RoleAPI.deleteByIds(ids, RepositoryAdapters.wrap(callback));
    }

    public void updateRoleStatus(long id, EnableStatus status, RepositoryCallback<Void> callback) {
        RoleAPI.updateStatus(id, status, RepositoryAdapters.wrap(callback));
    }

    public void getRoleMenuIds(int roleId, RepositoryCallback<List<Integer>> callback) {
        RoleAPI.getRoleMenuIds(roleId, RepositoryAdapters.wrap(callback));
    }

    public void updateRoleMenus(int roleId, List<Integer> menuIds, RepositoryCallback<Void> callback) {
        RoleAPI.updateRoleMenus(roleId, menuIds, RepositoryAdapters.wrap(callback));
    }

    public void getMenuList(RepositoryCallback<List<MenuVO>> callback) {
        MenuQuery query = new MenuQuery();
        MenuAPI.getList(query, RepositoryAdapters.wrap(callback));
    }
}
