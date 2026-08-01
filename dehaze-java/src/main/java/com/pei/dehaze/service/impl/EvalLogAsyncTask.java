package com.pei.dehaze.service.impl;

import cn.hutool.json.JSONObject;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.model.entity.SysEvalLog;
import com.pei.dehaze.service.client.PythonAlgorithmClient;
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
    public void execute(Long logId, Long algorithmId, String predUrl, String gtUrl) {
        long startTime = System.currentTimeMillis();
        try {
            JSONObject result = pythonClient.evaluate(algorithmId, predUrl, gtUrl);
            Integer status = result.getInt("status");
            if (status != null && status == LogStatusEnum.PROCESSING.getValue()) {
                Long pythonLogId = result.getLong("logId");
                result = pollEvalTask(pythonLogId);
            }

            int elapsed = (int) (System.currentTimeMillis() - startTime);

            Integer finalStatus = result.getInt("status");
            if (finalStatus != null && finalStatus == LogStatusEnum.FAILED.getValue()) {
                SysEvalLog update = new SysEvalLog();
                update.setId(logId);
                update.setTime(elapsed);
                update.setStatus(LogStatusEnum.FAILED);
                update.setErrorMessage(result.getStr("errorMessage"));
                evalLogMapper.updateById(update);
                meterRegistry.counter("dehaze_evaluation_total", "status", "failure").increment();
                log.error("评估失败(Python): algorithmId={}, evalLogId={}, error={}",
                        algorithmId, logId, result.getStr("errorMessage"));
                return;
            }

            SysEvalLog update = new SysEvalLog();
            update.setId(logId);
            update.setTime(elapsed);
            update.setResult(result.getStr("metrics"));
            update.setStatus(LogStatusEnum.COMPLETED);
            evalLogMapper.updateById(update);

            meterRegistry.counter("dehaze_evaluation_total", "status", "success").increment();
            log.debug("评估完成: algorithmId={}, evalLogId={}, time={}ms",
                    algorithmId, logId, elapsed);
        } catch (Exception e) {
            int elapsed = (int) (System.currentTimeMillis() - startTime);
            SysEvalLog update = new SysEvalLog();
            update.setId(logId);
            update.setTime(elapsed);
            update.setStatus(LogStatusEnum.FAILED);
            update.setErrorMessage(e.getMessage());
            evalLogMapper.updateById(update);

            meterRegistry.counter("dehaze_evaluation_total", "status", "failure").increment();
            log.error("评估失败: algorithmId={}, evalLogId={}, error={}",
                    algorithmId, logId, e.getMessage(), e);
        }
    }

    private JSONObject pollEvalTask(Long pythonLogId) throws InterruptedException {
        long deadline = System.currentTimeMillis() + POLL_TIMEOUT_MS;
        while (System.currentTimeMillis() < deadline) {
            Thread.sleep(POLL_INTERVAL_MS);
            JSONObject result = pythonClient.getEvalTaskStatus(pythonLogId);
            Integer status = result.getInt("status");
            if (status != null && (status == LogStatusEnum.COMPLETED.getValue() || status == LogStatusEnum.FAILED.getValue())) {
                return result;
            }
        }
        throw new RuntimeException("Python 评估任务 " + pythonLogId + " 轮询超时");
    }
}
