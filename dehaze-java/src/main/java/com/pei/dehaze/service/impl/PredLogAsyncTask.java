package com.pei.dehaze.service.impl;

import cn.hutool.json.JSONObject;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.service.client.PythonAlgorithmClient;
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

    @Async("datasetTaskExecutor")
    public void execute(Long logId, Long algorithmId, String imageUrl, String params) {
        long startTime = System.currentTimeMillis();
        try {
            JSONObject result = pythonClient.predict(algorithmId, imageUrl, params);
            String status = result.getStr("status");
            if ("processing".equals(status)) {
                Long pythonLogId = result.getLong("logId");
                result = pollPredTask(pythonLogId);
            }

            int elapsed = (int) (System.currentTimeMillis() - startTime);

            if ("failed".equals(result.getStr("status"))) {
                SysPredLog update = new SysPredLog();
                update.setId(logId);
                update.setTime(elapsed);
                update.setStatus(LogStatusEnum.FAILED);
                update.setErrorMessage(result.getStr("errorMessage"));
                predLogMapper.updateById(update);
                log.error("预测失败(Python): algorithmId={}, predLogId={}, error={}",
                        algorithmId, logId, result.getStr("errorMessage"));
                return;
            }

            SysPredLog update = new SysPredLog();
            update.setId(logId);
            update.setTime(elapsed);
            update.setPredUrl(result.getStr("resultUrl"));
            update.setPredMd5(result.getStr("resultMd5"));
            update.setStatus(LogStatusEnum.COMPLETED);
            predLogMapper.updateById(update);

            log.info("预测完成: algorithmId={}, predLogId={}, time={}ms",
                    algorithmId, logId, elapsed);
        } catch (Exception e) {
            int elapsed = (int) (System.currentTimeMillis() - startTime);
            SysPredLog update = new SysPredLog();
            update.setId(logId);
            update.setTime(elapsed);
            update.setStatus(LogStatusEnum.FAILED);
            update.setErrorMessage(e.getMessage());
            predLogMapper.updateById(update);

            log.error("预测失败: algorithmId={}, predLogId={}, error={}",
                    algorithmId, logId, e.getMessage(), e);
        }
    }

    private JSONObject pollPredTask(Long pythonLogId) throws InterruptedException {
        long deadline = System.currentTimeMillis() + POLL_TIMEOUT_MS;
        while (System.currentTimeMillis() < deadline) {
            Thread.sleep(POLL_INTERVAL_MS);
            JSONObject result = pythonClient.getPredTaskStatus(pythonLogId);
            String status = result.getStr("status");
            if ("completed".equals(status) || "failed".equals(status)) {
                return result;
            }
        }
        throw new RuntimeException("Python 预测任务 " + pythonLogId + " 轮询超时");
    }
}
