package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.evaluation.EvalParam;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.evaluation.EvalTaskStatus;
import com.pei.dehaze.sdk.model.evaluation.EvaluationLogVO;
import com.pei.dehaze.sdk.network.ApiException;
import com.pei.dehaze.sdk.network.ResultUnwrapper;
import com.pei.dehaze.sdk.service.ModelApiService;

import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import retrofit2.Call;
import retrofit2.Response;

/**
 * 评估API接口封装
 */
public class ModelAPI {

    private static final int DEFAULT_INTERVAL_MS = 2000;
    private static final int DEFAULT_TIMEOUT_MS = 120000;
    private static final ScheduledExecutorService POLL_EXECUTOR =
            Executors.newScheduledThreadPool(2, r -> {
                Thread t = new Thread(r, "ModelAPI-Poll");
                t.setDaemon(true);
                return t;
            });

    private ModelAPI() {
    }

    /**
     * 提交效果评估（POST，立即返回 logId + status）
     */
    public static void evaluate(EvalParam data, ApiCallback<EvalResult> callback) {
        Call<Result<EvalResult>> call = DehazeSDK.getInstance().getModelApiService().evaluate(data);
        call.enqueue(callback);
    }

    /**
     * 查询评估任务状态
     */
    public static void getEvalTaskStatus(long taskId, ApiCallback<EvalResult> callback) {
        Call<Result<EvalResult>> call = DehazeSDK.getInstance().getModelApiService().getEvalTaskStatus(taskId);
        call.enqueue(callback);
    }

    /**
     * 提交评估并轮询直至终态（completed/failed）或超时
     */
    public static void evaluateAndWait(EvalParam data, ApiCallback<EvalResult> callback) {
        evaluateAndWait(data, DEFAULT_INTERVAL_MS, DEFAULT_TIMEOUT_MS, callback);
    }

    public static void evaluateAndWait(EvalParam data, int intervalMs, int timeoutMs,
                                       ApiCallback<EvalResult> callback) {
        evaluate(data, new ApiCallback<EvalResult>() {
            @Override
            public void onSuccess(EvalResult result) {
                if (result == null || result.getStatus() == null
                        || result.getStatus() != EvalTaskStatus.PROCESSING) {
                    callback.onSuccess(result);
                    return;
                }
                pollEvalTask(result.getLogId(), intervalMs, timeoutMs, callback);
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

    private static void pollEvalTask(Long logId, int intervalMs, int timeoutMs,
                                     ApiCallback<EvalResult> callback) {
        if (logId == null) {
            callback.onFailure(new ApiException(0, "评估任务 logId 为空"));
            return;
        }
        long deadline = System.currentTimeMillis() + timeoutMs;
        ModelApiService service = DehazeSDK.getInstance().getModelApiService();
        POLL_EXECUTOR.schedule(() -> {
            try {
                while (System.currentTimeMillis() < deadline) {
                    Thread.sleep(intervalMs);
                    Call<Result<EvalResult>> call = service.getEvalTaskStatus(logId);
                    Response<Result<EvalResult>> resp = call.execute();
                    EvalResult result = ResultUnwrapper.unwrap(resp);
                    if (result == null || result.getStatus() == null) {
                        callback.onFailure(new ApiException(0, "查询评估任务状态失败"));
                        return;
                    }
                    if (result.getStatus() != EvalTaskStatus.PROCESSING) {
                        callback.onSuccess(result);
                        return;
                    }
                }
                callback.onFailure(new ApiException(0, "评估任务 " + logId + " 超时（" + timeoutMs + "ms）"));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                callback.onFailure(new ApiException(0, "评估任务轮询被中断"));
            } catch (IOException e) {
                callback.onFailure(new ApiException(0, e.getMessage()));
            } catch (ApiException e) {
                callback.onFailure(e);
            }
        }, 0, TimeUnit.MILLISECONDS);
    }

    /**
     * 分页查询评估日志
     */
    public static void listEvaluationLogs(String modelId, int pageNum, int pageSize,
                                          ApiCallback<PageResult<EvaluationLogVO>> callback) {
        Call<Result<PageResult<EvaluationLogVO>>> call = DehazeSDK.getInstance().getModelApiService()
                .listEvaluationLogs(modelId, pageNum, pageSize);
        call.enqueue(callback);
    }

}
