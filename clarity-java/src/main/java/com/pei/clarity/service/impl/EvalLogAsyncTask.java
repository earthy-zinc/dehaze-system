package com.pei.clarity.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.json.JSONObject;
import com.pei.clarity.common.enums.LogStatusEnum;
import com.pei.clarity.mapper.SysEvalLogMapper;
import com.pei.clarity.model.entity.SysEvalLog;
import com.pei.clarity.service.client.PythonAlgorithmClient;
import io.micrometer.core.instrument.MeterRegistry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
public class EvalLogAsyncTask {

    private static final long POLL_INTERVAL_MS = 2000;
    private static final long POLL_TIMEOUT_MS = 300000;

    private final SysEvalLogMapper evalLogMapper;
    private final PythonAlgorithmClient pythonClient;
    private final MeterRegistry meterRegistry;

    @Async("datasetTaskExecutor")
    public void execute(Long logId, String modelId, String predUrl, String gtUrl) {
        long startTime = System.currentTimeMillis();
        try {
            JSONObject result = pythonClient.evaluate(modelId, predUrl, gtUrl);
            Integer status = result.getInt("status");
            if (status != null && status == LogStatusEnum.PROCESSING.getValue()) {
                Long pythonLogId = result.getLong("logId");
                result = pollEvalTask(pythonLogId);
            }

            int elapsedMs = (int) (System.currentTimeMillis() - startTime);

            Integer finalStatus = result.getInt("status");
            if (finalStatus != null && finalStatus == LogStatusEnum.FAILED.getValue()) {
                SysEvalLog update = new SysEvalLog();
                update.setId(logId);
                update.setTime(elapsedMs / 1000);
                update.setStatus(LogStatusEnum.FAILED);
                update.setErrorMessage(result.getStr("errorMessage"));
                evalLogMapper.updateById(update);
                meterRegistry
                        .counter("dehaze_evaluation_total", "status", "failure")
                        .increment();
                log.error(
                        "评估失败(Python): modelId={}, evalLogId={}, error={}",
                        modelId,
                        logId,
                        result.getStr("errorMessage"));
                return;
            }

            SysEvalLog update = new SysEvalLog();
            update.setId(logId);
            update.setTime(elapsedMs / 1000);
            update.setResult(result.getStr("metrics"));
            // pred_md5/gt_md5 是图片内容 MD5，由算法服务下载图片后返回；仅 URL 入参时本地算不出，不得用 URL 的 MD5 冒充
            String predMd5 = result.getStr("predMd5");
            if (CharSequenceUtil.isNotBlank(predMd5)) {
                update.setPredMd5(predMd5);
            }
            String gtMd5 = result.getStr("gtMd5");
            if (CharSequenceUtil.isNotBlank(gtMd5)) {
                update.setGtMd5(gtMd5);
            }
            update.setStatus(LogStatusEnum.COMPLETED);
            evalLogMapper.updateById(update);

            meterRegistry
                    .counter("dehaze_evaluation_total", "status", "success")
                    .increment();
            log.debug("评估完成: modelId={}, evalLogId={}, time={}ms", modelId, logId, elapsedMs);
        } catch (Exception e) {
            int elapsedMs = (int) (System.currentTimeMillis() - startTime);
            SysEvalLog update = new SysEvalLog();
            update.setId(logId);
            update.setTime(elapsedMs / 1000);
            update.setStatus(LogStatusEnum.FAILED);
            update.setErrorMessage(e.getMessage());
            evalLogMapper.updateById(update);

            meterRegistry
                    .counter("dehaze_evaluation_total", "status", "failure")
                    .increment();
            log.error("评估失败: modelId={}, evalLogId={}, error={}", modelId, logId, e.getMessage(), e);
        }
    }

    private JSONObject pollEvalTask(Long pythonLogId) throws InterruptedException {
        long deadline = System.currentTimeMillis() + POLL_TIMEOUT_MS;
        while (System.currentTimeMillis() < deadline) {
            Thread.sleep(POLL_INTERVAL_MS);
            JSONObject result = pythonClient.getEvalTaskStatus(pythonLogId);
            Integer status = result.getInt("status");
            if (status != null
                    && (status == LogStatusEnum.COMPLETED.getValue() || status == LogStatusEnum.FAILED.getValue())) {
                return result;
            }
        }
        throw new RuntimeException("Python 评估任务 " + pythonLogId + " 轮询超时");
    }
}
