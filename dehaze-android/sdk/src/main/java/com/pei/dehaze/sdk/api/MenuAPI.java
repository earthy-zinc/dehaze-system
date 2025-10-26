package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.menu.MenuQuery;
import com.pei.dehaze.sdk.model.menu.MenuVO;
import com.pei.dehaze.sdk.model.menu.MenuForm;
import com.pei.dehaze.sdk.model.menu.RouteVO;

import java.util.List;

import retrofit2.Call;

/**
 * 菜单相关API接口封装
 */
public class MenuAPI {

    /**
     * 获取路由列表
     *
     * @param callback 回调函数
     */
    public static void getRoutes(ApiCallback<List<RouteVO>> callback) {
        Call<Result<List<RouteVO>>> call = DehazeSDK.getInstance().getMenuApiService().getRoutes();
        call.enqueue(callback);
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
    public static void getFormData(int id, ApiCallback<MenuForm> callback) {
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
    public static void update(String id, MenuForm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getMenuApiService().updateMenu(id, data);
        call.enqueue(callback);
    }

    /**
     * 删除菜单
     *
     * @param id       菜单ID
     * @param callback 回调函数
     */
    public static void deleteById(int id, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getMenuApiService().deleteMenu(id);
        call.enqueue(callback);
    }
}