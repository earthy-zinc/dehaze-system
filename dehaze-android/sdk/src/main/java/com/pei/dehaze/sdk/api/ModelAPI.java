package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.evaluation.EvalParam;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.evaluation.EvaluationLogVO;
import com.pei.dehaze.sdk.model.prediction.PredEvalTaskStatus;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;
import com.pei.dehaze.sdk.model.prediction.PredictionQuota;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.sdk.network.ApiException;
import com.pei.dehaze.sdk.service.ModelApiService;

import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import retrofit2.Call;
import retrofit2.Response;

/**
 * 预测/评估相关API接口封装
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
     * 执行去雾预测（POST 提交，立即返回 logId + status）
     */
    public static void predict(PredParam data, ApiCallback<PredResult> callback) {
        Call<Result<PredResult>> call = DehazeSDK.getInstance().getModelApiService().predict(data);
        call.enqueue(callback);
    }

    /**
     * 查询预测任务状态
     */
    public static void getPredTaskStatus(long taskId, ApiCallback<PredResult> callback) {
        Call<Result<PredResult>> call = DehazeSDK.getInstance().getModelApiService().getPredTaskStatus(taskId);
        call.enqueue(callback);
    }

    /**
     * 提交预测并轮询直至终态（completed/failed）或超时
     *
     * @param data     预测参数
     * @param callback 回调
     */
    public static void predictAndWait(PredParam data, ApiCallback<PredResult> callback) {
        predictAndWait(data, DEFAULT_INTERVAL_MS, DEFAULT_TIMEOUT_MS, callback);
    }

    public static void predictAndWait(PredParam data, int intervalMs, int timeoutMs,
                                      ApiCallback<PredResult> callback) {
        predict(data, new ApiCallback<PredResult>() {
            @Override
            public void onSuccess(PredResult result) {
                if (result == null || result.getStatus() == null) {
                    callback.onSuccess(result);
                    return;
                }
                if (result.getStatus() != PredEvalTaskStatus.PROCESSING) {
                    callback.onSuccess(result);
                    return;
                }
                pollPredTask(result.getLogId(), intervalMs, timeoutMs, callback);
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

    private static void pollPredTask(Long logId, int intervalMs, int timeoutMs,
                                     ApiCallback<PredResult> callback) {
        if (logId == null) {
            callback.onFailure(new ApiException(0, "预测任务 logId 为空"));
            return;
        }
        long deadline = System.currentTimeMillis() + timeoutMs;
        ModelApiService service = DehazeSDK.getInstance().getModelApiService();
        POLL_EXECUTOR.schedule(() -> {
            try {
                while (System.currentTimeMillis() < deadline) {
                    Thread.sleep(intervalMs);
                    Call<Result<PredResult>> call = service.getPredTaskStatus(logId);
                    Response<Result<PredResult>> resp = call.execute();
                    PredResult result = unwrap(resp);
                    if (result == null || result.getStatus() == null) {
                        callback.onFailure(new ApiException(0, "查询预测任务状态失败"));
                        return;
                    }
                    if (result.getStatus() == PredEvalTaskStatus.COMPLETED
                            || result.getStatus() == PredEvalTaskStatus.FAILED) {
                        callback.onSuccess(result);
                        return;
                    }
                }
                callback.onFailure(new ApiException(0, "预测任务 " + logId + " 超时（" + timeoutMs + "ms）"));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                callback.onFailure(new ApiException(0, "预测任务轮询被中断"));
            } catch (IOException e) {
                callback.onFailure(new ApiException(0, e.getMessage()));
            } catch (ApiException e) {
                callback.onFailure(e);
            }
        }, 0, TimeUnit.MILLISECONDS);
    }

    /**
     * 分页查询预测日志
     */
    public static void listPredictionLogs(Long algorithmId, int pageNum, int pageSize,
                                          ApiCallback<PageResult<PredictionLogVO>> callback) {
        Call<Result<PageResult<PredictionLogVO>>> call = DehazeSDK.getInstance().getModelApiService()
                .listPredictionLogs(algorithmId, pageNum, pageSize);
        call.enqueue(callback);
    }

    /**
     * 执行效果评估（POST 提交，立即返回 logId + status）
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
                if (result == null || result.getStatus() == null) {
                    callback.onSuccess(result);
                    return;
                }
                if (result.getStatus() != PredEvalTaskStatus.PROCESSING) {
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
                    EvalResult result = unwrap(resp);
                    if (result == null || result.getStatus() == null) {
                        callback.onFailure(new ApiException(0, "查询评估任务状态失败"));
                        return;
                    }
                    if (result.getStatus() == PredEvalTaskStatus.COMPLETED
                            || result.getStatus() == PredEvalTaskStatus.FAILED) {
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
    public static void listEvaluationLogs(Long algorithmId, int pageNum, int pageSize,
                                          ApiCallback<PageResult<EvaluationLogVO>> callback) {
        Call<Result<PageResult<EvaluationLogVO>>> call = DehazeSDK.getInstance().getModelApiService()
                .listEvaluationLogs(algorithmId, pageNum, pageSize);
        call.enqueue(callback);
    }

    /**
     * 查询 VIP 配额（剩余处理次数）
     */
    public static void getQuota(ApiCallback<PredictionQuota> callback) {
        Call<Result<PredictionQuota>> call = DehazeSDK.getInstance().getModelApiService().getQuota();
        call.enqueue(callback);
    }

    /**
     * 同步执行 HTTP 请求并解包 Result 容器，失败时抛出 ApiException
     */
    private static <T> T unwrap(Response<Result<T>> response) throws ApiException, IOException {
        if (response == null || !response.isSuccessful()) {
            int code = response == null ? -1 : response.code();
            throw new ApiException(code, "HTTP " + code);
        }
        Result<T> body = response.body();
        if (body == null) {
            throw new ApiException(0, "响应体为空");
        }
        if (!body.isSuccess()) {
            throw new ApiException(0, body.getCode(), body.getMsg());
        }
        return body.getData();
    }
}
