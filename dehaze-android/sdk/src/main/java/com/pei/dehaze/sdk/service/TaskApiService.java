package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.task.TaskCreateForm;
import com.pei.dehaze.sdk.model.task.TaskVO;

import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;
import retrofit2.http.Query;
import retrofit2.http.Streaming;

/**
 * 任务管理API服务接口
 * 对齐后端路由：/api/v1/tasks
 */
public interface TaskApiService {
    /**
     * 任务分页列表
     * GET /api/v1/tasks
     */
    @GET("/api/v1/tasks")
    Call<Result<PageResult<TaskVO>>> getTaskPage(
            @Query("status") String status,
            @Query("taskType") String taskType,
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize);

    /**
     * 创建任务
     * POST /api/v1/tasks
     */
    @POST("/api/v1/tasks")
    Call<Result<TaskVO>> createTask(@Body TaskCreateForm form);

    /**
     * 任务详情
     * GET /api/v1/tasks/{taskId}
     */
    @GET("/api/v1/tasks/{taskId}")
    Call<Result<TaskVO>> getTask(@Path("taskId") String taskId);

    /**
     * 下载导出文件（302重定向到文件存储）
     * GET /api/v1/tasks/{taskId}/download
     */
    @Streaming
    @GET("/api/v1/tasks/{taskId}/download")
    Call<ResponseBody> downloadTaskFile(@Path("taskId") String taskId);

    /**
     * 取消任务
     * POST /api/v1/tasks/{taskId}/cancel
     */
    @POST("/api/v1/tasks/{taskId}/cancel")
    Call<Result<Void>> cancelTask(@Path("taskId") String taskId);

    /**
     * 删除任务（Go 后端）
     * DELETE /api/v1/tasks/{id}
     */
    @retrofit2.http.DELETE("/api/v1/tasks/{id}")
    Call<Result<Void>> deleteTask(@Path("id") long id);
}
