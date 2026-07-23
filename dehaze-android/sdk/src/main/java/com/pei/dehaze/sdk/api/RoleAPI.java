package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.EnableStatus;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.role.RoleForm;
import com.pei.dehaze.sdk.model.role.RolePageVO;
import com.pei.dehaze.sdk.model.role.RoleQuery;
import retrofit2.Call;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 角色相关API接口封装
 */
public class RoleAPI {

    private RoleAPI() {
    }

    /**
     * 获取角色分页数据
     *
     * @param queryParams 查询参数
     * @param callback    回调函数
     */
    public static void getPage(RoleQuery queryParams, ApiCallback<PageResult<RolePageVO>> callback) {
        Call<Result<PageResult<RolePageVO>>> call = DehazeSDK.getInstance().getRoleApiService()
                .getRolePage(queryParams.getPageNum(), queryParams.getPageSize(), queryParams.getKeywords());
        call.enqueue(callback);
    }

    /**
     * 获取角色下拉数据源
     *
     * @param queryParams 查询参数
     * @param callback    回调函数
     */
    public static void getOptions(RoleQuery queryParams, ApiCallback<List<Option>> callback) {
        Call<Result<List<Option>>> call = DehazeSDK.getInstance().getRoleApiService()
                .getRoleOptions(queryParams.getPageNum(), queryParams.getPageSize(), queryParams.getKeywords());
        call.enqueue(callback);
    }

    /**
     * 获取角色的菜单ID集合
     *
     * @param roleId   角色ID
     * @param callback 回调函数
     */
    public static void getRoleMenuIds(int roleId, ApiCallback<List<Integer>> callback) {
        Call<Result<List<Integer>>> call = DehazeSDK.getInstance().getRoleApiService().getRoleMenuIds(roleId);
        call.enqueue(callback);
    }

    /**
     * 分配菜单权限给角色
     *
     * @param roleId   角色ID
     * @param data     菜单ID列表
     * @param callback 回调函数
     */
    public static void updateRoleMenus(int roleId, List<Integer> data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getRoleApiService().updateRoleMenus(roleId, data);
        call.enqueue(callback);
    }

    /**
     * 获取角色表单数据
     *
     * @param id       角色ID
     * @param callback 回调函数
     */
    public static void getFormData(int id, ApiCallback<RoleForm> callback) {
        Call<Result<RoleForm>> call = DehazeSDK.getInstance().getRoleApiService().getRoleFormData(id);
        call.enqueue(callback);
    }

    /**
     * 添加角色
     *
     * @param data     角色表单数据
     * @param callback 回调函数
     */
    public static void add(RoleForm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getRoleApiService().addRole(data);
        call.enqueue(callback);
    }

    /**
     * 更新角色
     *
     * @param id       角色ID
     * @param data     角色表单数据
     * @param callback 回调函数
     */
    public static void update(int id, RoleForm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getRoleApiService().updateRole(id, data);
        call.enqueue(callback);
    }

    /**
     * 批量删除角色，多个以英文逗号(,)分割
     *
     * @param ids      角色ID列表
     * @param callback 回调函数
     */
    public static void deleteByIds(List<Long> ids, ApiCallback<Void> callback) {
        String joined = ids.stream().map(String::valueOf).collect(Collectors.joining(","));
        Call<Result<Void>> call = DehazeSDK.getInstance().getRoleApiService().deleteRoles(joined);
        call.enqueue(callback);
    }

    /**
     * 修改角色状态
     *
     * @param id       角色ID
     * @param status   状态
     * @param callback 回调函数
     */
    public static void updateStatus(long id, EnableStatus status, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getRoleApiService().updateRoleStatus(id, status.getValue());
        call.enqueue(callback);
    }
}
