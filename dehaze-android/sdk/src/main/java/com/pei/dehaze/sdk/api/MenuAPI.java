package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.EnableStatus;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.menu.MenuQuery;
import com.pei.dehaze.sdk.model.menu.MenuForm;
import com.pei.dehaze.sdk.model.menu.MenuVO;

import java.util.List;
import java.util.stream.Collectors;

import retrofit2.Call;

/**
 * 菜单相关API接口封装
 */
public class MenuAPI {

    private MenuAPI() {
    }

    /**
     * 获取菜单树形列表
     *
     * @param queryParams 查询参数
     * @param callback    回调函数
     */
    public static void getList(MenuQuery queryParams, ApiCallback<List<MenuVO>> callback) {
        Call<Result<List<MenuVO>>> call = DehazeSDK.getInstance().getMenuApiService().getMenuList(queryParams.getKeywords());
        call.enqueue(callback);
    }

    /**
     * 获取菜单下拉数据源
     *
     * @param callback 回调函数
     */
    public static void getOptions(ApiCallback<List<Option>> callback) {
        Call<Result<List<Option>>> call = DehazeSDK.getInstance().getMenuApiService().getMenuOptions();
        call.enqueue(callback);
    }

    /**
     * 获取菜单表单数据
     *
     * @param id       菜单ID
     * @param callback 回调函数
     */
    public static void getFormData(long id, ApiCallback<MenuForm> callback) {
        Call<Result<MenuForm>> call = DehazeSDK.getInstance().getMenuApiService().getMenuFormData(id);
        call.enqueue(callback);
    }

    /**
     * 添加菜单
     *
     * @param data     菜单表单数据
     * @param callback 回调函数
     */
    public static void add(MenuForm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getMenuApiService().addMenu(data);
        call.enqueue(callback);
    }

    /**
     * 修改菜单
     *
     * @param id       菜单ID
     * @param data     菜单表单数据
     * @param callback 回调函数
     */
    public static void update(long id, MenuForm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getMenuApiService().updateMenu(id, data);
        call.enqueue(callback);
    }

    /**
     * 删除菜单（支持批量）
     *
     * @param ids      菜单ID列表
     * @param callback 回调函数
     */
    public static void deleteByIds(List<Long> ids, ApiCallback<Void> callback) {
        String joined = ids.stream().map(String::valueOf).collect(Collectors.joining(","));
        Call<Result<Void>> call = DehazeSDK.getInstance().getMenuApiService().deleteMenus(joined);
        call.enqueue(callback);
    }

    /**
     * 修改菜单显示状态
     *
     * @param id       菜单ID
     * @param visible   显示状态(1:显示;0:隐藏)
     * @param callback 回调函数
     */
    public static void updateVisible(long id, EnableStatus visible, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getMenuApiService().updateMenuVisible(id, visible.getValue());
        call.enqueue(callback);
    }
}