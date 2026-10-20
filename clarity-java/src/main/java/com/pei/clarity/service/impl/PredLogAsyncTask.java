package com.pei.clarity.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.json.JSONObject;
import com.pei.clarity.common.enums.LogStatusEnum;
import com.pei.clarity.mapper.SysPredLogMapper;
import com.pei.clarity.model.entity.SysPredLog;
import com.pei.clarity.service.client.PythonAlgorithmClient;
import io.micrometer.core.instrument.MeterRegistry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
public class PredLogAsyncTask {

    private static final long POLL_INTERVAL_MS = 2000;
    private static final long POLL_TIMEOUT_MS = 300000;

    private final SysPredLogMapper predLogMapper;
    private final PythonAlgorithmClient pythonClient;
    private final MeterRegistry meterRegistry;

    @Async("datasetTaskExecutor")
    public void execute(Long logId, String model, Long fileId, String imageUrl, String params) {
        long startTime = System.currentTimeMillis();
        try {
            JSONObject result = pythonClient.predict(model, fileId, imageUrl, params);
            Integer status = result.getInt("status");
            if (status != null && status == LogStatusEnum.PROCESSING.getValue()) {
                Long pythonTaskId = result.getLong("taskId");
                result = pollPredTask(pythonTaskId);
            }

            int elapsedMs = (int) (System.currentTimeMillis() - startTime);

            Integer finalStatus = result.getInt("status");
            if (finalStatus != null && finalStatus == LogStatusEnum.FAILED.getValue()) {
                SysPredLog update = new SysPredLog();
                update.setId(logId);
                update.setTime(elapsedMs / 1000);
                update.setStatus(LogStatusEnum.FAILED);
                update.setErrorMessage(result.getStr("errorMessage"));
                predLogMapper.updateById(update);
                meterRegistry
                        .counter("dehaze_prediction_total", "status", "failure")
                        .increment();
                log.error(
                        "图像任务失败(Python): model={}, predLogId={}, error={}",
                        model,
                        logId,
                        result.getStr("errorMessage"));
                return;
            }

            SysPredLog update = new SysPredLog();
            update.setId(logId);
            update.setTime(elapsedMs / 1000);
            update.setPredUrl(result.getStr("resultUrl"));
            update.setPredMd5(result.getStr("resultMd5"));
            // origin_md5 是原图内容 MD5，由算法服务下载图片后返回；仅 imageUrl 入参时本地算不出
            String originMd5 = result.getStr("originMd5");
            if (CharSequenceUtil.isNotBlank(originMd5)) {
                update.setOriginMd5(originMd5);
            }
            update.setStatus(LogStatusEnum.COMPLETED);
            predLogMapper.updateById(update);

            meterRegistry
                    .counter("dehaze_prediction_total", "status", "success")
                    .increment();
            log.debug("图像任务完成: model={}, predLogId={}, time={}ms", model, logId, elapsedMs);
        } catch (Exception e) {
            int elapsedMs = (int) (System.currentTimeMillis() - startTime);
            SysPredLog update = new SysPredLog();
            update.setId(logId);
            update.setTime(elapsedMs / 1000);
            update.setStatus(LogStatusEnum.FAILED);
            update.setErrorMessage(e.getMessage());
            predLogMapper.updateById(update);

            meterRegistry
                    .counter("dehaze_prediction_total", "status", "failure")
                    .increment();
            log.error("图像任务失败: model={}, predLogId={}, error={}", model, logId, e.getMessage(), e);
        }
    }

    private JSONObject pollPredTask(Long pythonTaskId) throws InterruptedException {
        long deadline = System.currentTimeMillis() + POLL_TIMEOUT_MS;
        while (System.currentTimeMillis() < deadline) {
            Thread.sleep(POLL_INTERVAL_MS);
            JSONObject result = pythonClient.getPredTaskStatus(pythonTaskId);
            Integer status = result.getInt("status");
            if (status != null
                    && (status == LogStatusEnum.COMPLETED.getValue() || status == LogStatusEnum.FAILED.getValue())) {
                return result;
            }
        }
        throw new RuntimeException("Python 图像任务 " + pythonTaskId + " 轮询超时");
    }
}
