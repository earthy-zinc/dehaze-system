package com.pei.clarity.sdk.service;

import com.pei.clarity.sdk.model.PageResult;
import com.pei.clarity.sdk.model.Result;
import com.pei.clarity.sdk.model.image_task.ImageEditForm;
import com.pei.clarity.sdk.model.image_task.ImageTaskResponse;
import com.pei.clarity.sdk.model.image_task.ImageTaskVO;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;
import retrofit2.http.Query;

/**
 * 图像任务API服务接口
 * 对齐后端路由：/api/v1/images（提交-轮询任务模式）
 */
public interface AiImageApiService {

    /**
     * 提交图像编辑任务（去雾等）
     * POST /api/v1/images/edits
     */
    @POST("/api/v1/images/edits")
    Call<Result<ImageTaskResponse>> submitEdit(@Body ImageEditForm form);

    /**
     * 查询图像任务状态
     * GET /api/v1/images/tasks/{taskId}
     */
    @GET("/api/v1/images/tasks/{taskId}")
    Call<Result<ImageTaskResponse>> getTask(@Path("taskId") long taskId);

    /**
     * 取消图像任务（幂等，仅处理中任务可取消）
     * POST /api/v1/images/tasks/{taskId}/cancel
     */
    @POST("/api/v1/images/tasks/{taskId}/cancel")
    Call<Result<ImageTaskResponse>> cancelTask(@Path("taskId") long taskId);

    /**
     * 图像任务历史分页列表
     * GET /api/v1/images/tasks
     */
    @GET("/api/v1/images/tasks")
    Call<Result<PageResult<ImageTaskVO>>> listTasks(
            @Query("model") String model,
            @Query("pageNum") int pageNum,
            @Query("pageSize") int pageSize);
}
