package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.dict.*;
import retrofit2.Call;

import java.util.List;

/**
 * 字典相关API接口封装
 */
public class DictAPI {

    /**
     * 字典类型分页列表
     *
     * @param queryParams 查询参数
     * @param callback    回调函数
     */
    public static void getDictTypePage(DictTypeQuery queryParams, ApiCallback<PageResult<DictTypePageVO>> callback) {
        Call<Result<PageResult<DictTypePageVO>>> call = DehazeSDK.getInstance().getDictApiService()
                .getDictTypePage(queryParams.getPageNum(), queryParams.getPageSize(), queryParams.getKeywords());
        call.enqueue(callback);
    }

    /**
     * 字典类型表单数据
     *
     * @param id       字典类型ID
     * @param callback 回调函数
     */
    public static void getDictTypeForm(int id, ApiCallback<DictTypeForm> callback) {
        Call<Result<DictTypeForm>> call = DehazeSDK.getInstance().getDictApiService().getDictTypeFormData(id);
        call.enqueue(callback);
    }

    /**
     * 新增字典类型
     *
     * @param data     字典类型表单数据
     * @param callback 回调函数
     */
    public static void addDictType(DictTypeForm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDictApiService().addDictType(data);
        call.enqueue(callback);
    }

    /**
     * 修改字典类型
     *
     * @param id       字典类型ID
     * @param data     字典类型表单数据
     * @param callback 回调函数
     */
    public static void updateDictType(int id, DictTypeForm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDictApiService().updateDictType(id, data);
        call.enqueue(callback);
    }

    /**
     * 删除字典类型
     *
     * @param ids      字典类型ID列表
     * @param callback 回调函数
     */
    public static void deleteDictTypes(String ids, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDictApiService().deleteDictTypes(ids);
        call.enqueue(callback);
    }

    /**
     * 获取字典类型的数据项
     *
     * @param id       字典类型ID
     * @param callback 回调函数
     */
    public static void getDictOptions(long id, ApiCallback<List<Option>> callback) {
        Call<Result<List<Option>>> call = DehazeSDK.getInstance().getDictApiService().getDictOptions(id);
        call.enqueue(callback);
    }

    /**
     * 字典分页列表
     *
     * @param queryParams 查询参数
     * @param callback    回调函数
     */
    public static void getDictPage(DictQuery queryParams, ApiCallback<PageResult<DictPageVO>> callback) {
        Call<Result<PageResult<DictPageVO>>> call = DehazeSDK.getInstance().getDictApiService()
                .getDictPage(queryParams.getPageNum(), queryParams.getPageSize(), queryParams.getName(), queryParams.getTypeCode());
        call.enqueue(callback);
    }

    /**
     * 获取字典表单数据
     *
     * @param id       字典ID
     * @param callback 回调函数
     */
    public static void getDictFormData(int id, ApiCallback<DictForm> callback) {
        Call<Result<DictForm>> call = DehazeSDK.getInstance().getDictApiService().getDictFormData(id);
        call.enqueue(callback);
    }

    /**
     * 新增字典
     *
     * @param data     字典表单数据
     * @param callback 回调函数
     */
    public static void addDict(DictForm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDictApiService().addDict(data);
        call.enqueue(callback);
    }

    /**
     * 修改字典项
     *
     * @param id       字典ID
     * @param data     字典表单数据
     * @param callback 回调函数
     */
    public static void updateDict(int id, DictForm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDictApiService().updateDict(id, data);
        call.enqueue(callback);
    }

    /**
     * 删除字典
     *
     * @param ids      字典ID列表
     * @param callback 回调函数
     */
    public static void deleteDictByIds(String ids, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getDictApiService().deleteDicts(ids);
        call.enqueue(callback);
    }
}
