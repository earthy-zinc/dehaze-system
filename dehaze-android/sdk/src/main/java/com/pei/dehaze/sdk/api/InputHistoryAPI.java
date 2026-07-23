package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.input_history.BatchDeleteForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryQuery;
import com.pei.dehaze.sdk.model.input_history.InputHistoryUpdateForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryVO;
import com.pei.dehaze.sdk.model.input_history.SyncResultVO;

import java.util.List;

import retrofit2.Call;

/**
 * 图像输入历史API接口封装
 */
public class InputHistoryAPI {

    private InputHistoryAPI() {
    }

    /**
     * 分页查询历史记录
     */
    public static void listHistory(InputHistoryQuery query, ApiCallback<PageResult<InputHistoryVO>> callback) {
        Call<Result<PageResult<InputHistoryVO>>> call = DehazeSDK.getInstance().getInputHistoryApiService().listHistory(
                query.getInputSource() != null ? query.getInputSource().getValue() : null,
                query.getFavoriteOnly(),
                query.getKeywords(),
                query.getPageNum(),
                query.getPageSize());
        call.enqueue(callback);
    }

    /**
     * 历史记录详情
     */
    public static void getHistory(long id, ApiCallback<InputHistoryVO> callback) {
        Call<Result<InputHistoryVO>> call = DehazeSDK.getInstance().getInputHistoryApiService().getHistory(id);
        call.enqueue(callback);
    }

    /**
     * 创建历史记录
     */
    public static void createHistory(InputHistoryForm form, ApiCallback<InputHistoryVO> callback) {
        Call<Result<InputHistoryVO>> call = DehazeSDK.getInstance().getInputHistoryApiService().createHistory(form);
        call.enqueue(callback);
    }

    /**
     * 更新历史记录（如收藏、补充处理结果）
     */
    public static void updateHistory(long id, InputHistoryUpdateForm form, ApiCallback<InputHistoryVO> callback) {
        Call<Result<InputHistoryVO>> call = DehazeSDK.getInstance().getInputHistoryApiService().updateHistory(id, form);
        call.enqueue(callback);
    }

    /**
     * 删除单条历史记录
     */
    public static void deleteHistory(long id, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getInputHistoryApiService().deleteHistory(id);
        call.enqueue(callback);
    }

    /**
     * 批量删除历史记录
     */
    public static void batchDeleteHistory(List<Long> ids, ApiCallback<Void> callback) {
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(ids);
        Call<Result<Void>> call = DehazeSDK.getInstance().getInputHistoryApiService().batchDeleteHistory(form);
        call.enqueue(callback);
    }

    /**
     * 清空历史记录
     */
    public static void clearHistory(ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getInputHistoryApiService().clearHistory();
        call.enqueue(callback);
    }

    /**
     * 同步本地与云端历史
     */
    public static void syncHistory(List<InputHistoryForm> items, ApiCallback<SyncResultVO> callback) {
        Call<Result<SyncResultVO>> call = DehazeSDK.getInstance().getInputHistoryApiService().syncHistory(items);
        call.enqueue(callback);
    }
}
