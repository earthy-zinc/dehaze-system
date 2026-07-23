package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.DeptAPI;
import com.pei.dehaze.sdk.api.RoleAPI;
import com.pei.dehaze.sdk.api.UserAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.EnableStatus;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.role.RoleQuery;
import com.pei.dehaze.sdk.model.user.UserForm;
import com.pei.dehaze.sdk.model.user.UserPageVO;
import com.pei.dehaze.sdk.model.user.UserQuery;

import java.io.File;
import java.util.List;

public class UserRepository {

    public void getUsers(UserQuery query, RepositoryCallback<PageResult<UserPageVO>> callback) {
        UserAPI.getPage(query, RepositoryAdapters.wrap(callback));
    }

    public void getUserForm(int userId, RepositoryCallback<UserForm> callback) {
        UserAPI.getFormData(userId, RepositoryAdapters.wrap(callback));
    }

    public void addUser(UserForm form, RepositoryCallback<Void> callback) {
        UserAPI.add(form, RepositoryAdapters.wrap(callback));
    }

    public void updateUser(int id, UserForm form, RepositoryCallback<Void> callback) {
        UserAPI.update(id, form, RepositoryAdapters.wrap(callback));
    }

    public void deleteUsers(List<Long> ids, RepositoryCallback<Void> callback) {
        UserAPI.deleteByIds(ids, RepositoryAdapters.wrap(callback));
    }

    public void updateUserPassword(int id, String password, RepositoryCallback<Void> callback) {
        UserAPI.updatePassword(id, password, RepositoryAdapters.wrap(callback));
    }

    public void updateUserStatus(long id, EnableStatus status, RepositoryCallback<Void> callback) {
        UserAPI.updateStatus(id, status, RepositoryAdapters.wrap(callback));
    }

    public void downloadTemplate(String filePath, RepositoryCallback<Void> callback) {
        UserAPI.downloadTemplate(filePath, RepositoryAdapters.wrap(callback));
    }

    public void exportUsers(UserQuery query, String filePath, RepositoryCallback<Void> callback) {
        UserAPI.export(query, filePath, RepositoryAdapters.wrap(callback));
    }

    public void importUsers(int deptId, File file, RepositoryCallback<Void> callback) {
        UserAPI.importUsers(deptId, file, RepositoryAdapters.wrap(callback));
    }

    public void getDeptOptions(RepositoryCallback<List<Option>> callback) {
        DeptAPI.getOptions(RepositoryAdapters.wrap(callback));
    }

    public void getRoleOptions(RepositoryCallback<List<Option>> callback) {
        RoleQuery query = new RoleQuery();
        RoleAPI.getOptions(query, RepositoryAdapters.wrap(callback));
    }
}
