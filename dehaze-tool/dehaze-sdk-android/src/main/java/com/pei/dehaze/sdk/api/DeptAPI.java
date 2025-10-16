package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.dept.DeptVO;
import com.pei.dehaze.sdk.model.dept.DeptQuery;
import com.pei.dehaze.sdk.model.dept.DeptForm;

import java.util.List;

import retrofit2.Call;

/**
 * 部门相关API接口封装
 */
public class DeptAPI {

    /**
     * 部门树形表格
     *
     * @param queryParams 查询参数
     * @param callback    回调函数
     */
    public static void getList(DeptQuery queryParams, ApiCallback<List<DeptVO>> callback) {
        Call<Result<List<DeptVO>>> call = DehazeSDK.getInstance().getDeptApiService()
                .getDeptList(queryParams.getKeywords(), queryParams.getStatus());
        call.enqueue(callback);
    }

    /**
     * 部门下拉列表
     *
     * @param callback 回调函数
     */
    public static void getOptions(ApiCallback<List<Option>> callback) {
        Call<Result<List<Option>>> call = DehazeSDK.getInstance().getDeptApiService().getDeptOptions();
        call.enqueue(callback);
    }

    /**
     * 获取部门详情
     *
     * @param id       部门ID
     * @param callback 回调函数
     */
    public static void getFormData(int id, ApiCallback<DeptForm> callback) {
        Call<Result<DeptForm>> call = DehazeSDK.getInstance().getDeptApiService().getDeptFormData(id);
        call.enqueue(callback);
    }

    /**
     * 新增部门
     *
     * @param data     部门表单数据
     * @param callback 回调函数
     */
    public static void add(DeptForm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDeptApiService().addDept(data);
        call.enqueue(callback);
    }

    /**
     * 修改部门
     *
     * @param id       部门ID
     * @param data     部门表单数据
     * @param callback 回调函数
     */
    public static void update(int id, DeptForm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDeptApiService().updateDept(id, data);
        call.enqueue(callback);
    }

    /**
     * 删除部门
     *
     * @param ids      部门ID列表
     * @param callback 回调函数
     */
    public static void deleteByIds(String ids, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDeptApiService().deleteDepts(ids);
        call.enqueue(callback);
    }
}