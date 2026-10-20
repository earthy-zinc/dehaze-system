package com.pei.clarity.sdk.api;

import com.pei.clarity.sdk.ApiCallback;
import com.pei.clarity.sdk.DehazeSDK;
import com.pei.clarity.sdk.model.PageResult;
import com.pei.clarity.sdk.model.Result;
import com.pei.clarity.sdk.model.image_task.ImageEditForm;
import com.pei.clarity.sdk.model.image_task.ImageTaskQuery;
import com.pei.clarity.sdk.model.image_task.ImageTaskResponse;
import com.pei.clarity.sdk.model.image_task.ImageTaskStatus;
import com.pei.clarity.sdk.model.image_task.ImageTaskVO;
import com.pei.clarity.sdk.network.ApiException;
import com.pei.clarity.sdk.network.ResultUnwrapper;
import com.pei.clarity.sdk.service.AiImageApiService;

import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import retrofit2.Call;

/**
 * 图像任务API接口封装（提交-轮询-取消-历史）
 */
public class AiImageAPI {

    private static final int DEFAULT_INTERVAL_MS = 2000;
    private static final int DEFAULT_TIMEOUT_MS = 120000;
    private static final ScheduledExecutorService POLL_EXECUTOR =
            Executors.newScheduledThreadPool(2, r -> {
                Thread t = new Thread(r, "AiImageAPI-Poll");
                t.setDaemon(true);
                return t;
            });

    private AiImageAPI() {
    }

    /**
     * 提交图像编辑任务（立即返回 taskId + status，缓存命中时直接 completed）
     */
    public static void submitImageTask(ImageEditForm data, ApiCallback<ImageTaskResponse> callback) {
        Call<Result<ImageTaskResponse>> call = DehazeSDK.getInstance().getAiImageApiService().submitEdit(data);
        call.enqueue(callback);
    }

    /**
     * 查询图像任务状态
     */
    public static void getImageTask(long taskId, ApiCallback<ImageTaskResponse> callback) {
        Call<Result<ImageTaskResponse>> call = DehazeSDK.getInstance().getAiImageApiService().getTask(taskId);
        call.enqueue(callback);
    }

    /**
     * 取消图像任务（幂等，仅处理中任务可取消）
     */
    public static void cancelImageTask(long taskId, ApiCallback<ImageTaskResponse> callback) {
        Call<Result<ImageTaskResponse>> call = DehazeSDK.getInstance().getAiImageApiService().cancelTask(taskId);
        call.enqueue(callback);
    }

    /**
     * 图像任务历史分页列表
     */
    public static void listImageTasks(ImageTaskQuery query, ApiCallback<PageResult<ImageTaskVO>> callback) {
        String model = query == null ? null : query.getModel();
        int pageNum = query == null ? 1 : query.getPageNum();
        int pageSize = query == null ? 10 : query.getPageSize();
        Call<Result<PageResult<ImageTaskVO>>> call = DehazeSDK.getInstance().getAiImageApiService()
                .listTasks(model, pageNum, pageSize);
        call.enqueue(callback);
    }

    /**
     * 提交图像任务并轮询至终态（completed/failed/canceled）或超时
     */
    public static void submitAndWait(ImageEditForm data, ApiCallback<ImageTaskResponse> callback) {
        submitAndWait(data, DEFAULT_INTERVAL_MS, DEFAULT_TIMEOUT_MS, callback);
    }

    public static void submitAndWait(ImageEditForm data, int intervalMs, int timeoutMs,
                                     ApiCallback<ImageTaskResponse> callback) {
        submitImageTask(data, new ApiCallback<ImageTaskResponse>() {
            @Override
            public void onSuccess(ImageTaskResponse result) {
                if (result == null || result.getStatus() == null
                        || result.getStatus() != ImageTaskStatus.PROCESSING) {
                    callback.onSuccess(result);
                    return;
                }
                pollTask(result.getTaskId(), intervalMs, timeoutMs, callback);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError(code, message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onFailure(e);
            }
        });
    }

    private static void pollTask(Long taskId, int intervalMs, int timeoutMs,
                                 ApiCallback<ImageTaskResponse> callback) {
        if (taskId == null) {
            callback.onFailure(new ApiException(0, "图像任务 taskId 为空"));
            return;
        }
        long deadline = System.currentTimeMillis() + timeoutMs;
        AiImageApiService service = DehazeSDK.getInstance().getAiImageApiService();
        POLL_EXECUTOR.schedule(() -> {
            try {
                while (System.currentTimeMillis() < deadline) {
                    Thread.sleep(intervalMs);
                    Call<Result<ImageTaskResponse>> call = service.getTask(taskId);
                    ImageTaskResponse result = ResultUnwrapper.unwrap(call.execute());
                    if (result == null || result.getStatus() == null) {
                        callback.onFailure(new ApiException(0, "查询图像任务状态失败"));
                        return;
                    }
                    if (result.getStatus() != ImageTaskStatus.PROCESSING) {
                        callback.onSuccess(result);
                        return;
                    }
                }
                callback.onFailure(new ApiException(0, "图像任务 " + taskId + " 超时（" + timeoutMs + "ms）"));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                callback.onFailure(new ApiException(0, "图像任务轮询被中断"));
            } catch (IOException e) {
                callback.onFailure(new ApiException(0, e.getMessage()));
            } catch (ApiException e) {
                callback.onFailure(e);
            }
        }, 0, TimeUnit.MILLISECONDS);
    }

}
