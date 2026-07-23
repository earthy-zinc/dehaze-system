package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.task.TaskCreateForm;
import com.pei.dehaze.sdk.model.task.TaskQuery;
import com.pei.dehaze.sdk.model.task.TaskVO;

import okhttp3.ResponseBody;
import retrofit2.Call;

/**
 * 任务管理API接口封装
 */
public class TaskAPI {

    private TaskAPI() {
    }

    /**
     * 任务分页列表
     */
    public static void getTaskPage(TaskQuery query, ApiCallback<PageResult<TaskVO>> callback) {
        Call<Result<PageResult<TaskVO>>> call = DehazeSDK.getInstance().getTaskApiService().getTaskPage(
                query.getStatus() != null ? query.getStatus().getValue() : null,
                query.getTaskType() != null ? query.getTaskType().getValue() : null,
                query.getPageNum(),
                query.getPageSize());
        call.enqueue(callback);
    }

    /**
     * 创建任务
     */
    public static void createTask(TaskCreateForm form, ApiCallback<TaskVO> callback) {
        Call<Result<TaskVO>> call = DehazeSDK.getInstance().getTaskApiService().createTask(form);
        call.enqueue(callback);
    }

    /**
     * 任务详情
     */
    public static void getTask(String taskId, ApiCallback<TaskVO> callback) {
        Call<Result<TaskVO>> call = DehazeSDK.getInstance().getTaskApiService().getTask(taskId);
        call.enqueue(callback);
    }

    /**
     * 下载导出文件（保存到本地路径）
     *
     * @param taskId    任务ID
     * @param savePath  保存路径
     * @param callback  回调
     */
    public static void downloadTaskFile(String taskId, String savePath, ApiCallback<Void> callback) {
        Call<ResponseBody> call = DehazeSDK.getInstance().getTaskApiService().downloadTaskFile(taskId);
        FileAPI.enqueueFileDownload(call, savePath, callback);
    }

    /**
     * 取消任务
     */
    public static void cancelTask(String taskId, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getTaskApiService().cancelTask(taskId);
        call.enqueue(callback);
    }
}
