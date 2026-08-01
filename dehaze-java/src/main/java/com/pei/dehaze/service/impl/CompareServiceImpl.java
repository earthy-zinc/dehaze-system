package com.pei.dehaze.service.impl;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysEvalLog;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.model.form.CompareReportForm;
import com.pei.dehaze.model.vo.CompareReportResultVO;
import com.pei.dehaze.service.CompareService;
import com.pei.dehaze.service.SysAlgorithmService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Slf4j
@Service
@RequiredArgsConstructor
public class CompareServiceImpl implements CompareService {

    private static final DateTimeFormatter DT_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final SysPredLogMapper predLogMapper;
    private final SysEvalLogMapper evalLogMapper;
    private final SysAlgorithmService algorithmService;

    @Override
    public CompareReportResultVO generateReport(CompareReportForm form) {
        SysPredLog predLog = predLogMapper.selectById(form.getLogId());
        if (predLog == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "处理记录不存在");
        }
        if (predLog.getStatus() != LogStatusEnum.COMPLETED) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "处理任务尚未完成，无法生成报告");
        }

        SysEvalLog reportTask = new SysEvalLog();
        reportTask.setAlgorithmId(predLog.getAlgorithmId());
        reportTask.setPredUrl(predLog.getOriginUrl());
        reportTask.setGtUrl(predLog.getPredUrl());
        reportTask.setStatus(LogStatusEnum.PROCESSING);
        reportTask.setResult(JSONUtil.createObj()
                .set("logId", form.getLogId())
                .set("format", form.getFormat())
                .set("includeMetrics", form.getIncludeMetrics())
                .set("includeFilters", form.getIncludeFilters())
                .toString());
        evalLogMapper.insert(reportTask);

        generateReportAsync(reportTask.getId(), predLog);

        CompareReportResultVO vo = new CompareReportResultVO();
        vo.setTaskId(reportTask.getId());
        vo.setStatus(LogStatusEnum.PROCESSING);
        return vo;
    }

    @Override
    public String getReportDownloadUrl(Long taskId) {
        SysEvalLog reportTask = evalLogMapper.selectById(taskId);
        if (reportTask == null || reportTask.getStatus() != LogStatusEnum.COMPLETED) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "报告不存在或尚未生成完成");
        }
        // 返回报告HTML内容，前端通过此URL获取
        return reportTask.getResult();
    }

    @Async("datasetTaskExecutor")
    public void generateReportAsync(Long taskId, SysPredLog predLog) {
        try {
            String algorithmName = "未知算法";
            if (predLog.getAlgorithmId() != null) {
                var algo = algorithmService.getById(predLog.getAlgorithmId());
                if (algo != null) {
                    algorithmName = algo.getName();
                }
            }

            String html = buildReportHtml(predLog, algorithmName);

            JSONObject result = JSONUtil.createObj()
                    .set("reportHtml", html)
                    .set("generatedAt", LocalDateTime.now().format(DT_FMT));

            SysEvalLog update = new SysEvalLog();
            update.setId(taskId);
            update.setStatus(LogStatusEnum.COMPLETED);
            update.setResult(result.toString());
            update.setTime((int) (predLog.getTime() != null ? predLog.getTime() : 0));
            evalLogMapper.updateById(update);

            log.info("对比报告生成完成: taskId={}", taskId);
        } catch (Exception e) {
            log.error("对比报告生成失败: taskId={}, error={}", taskId, e.getMessage(), e);
            SysEvalLog update = new SysEvalLog();
            update.setId(taskId);
            update.setStatus(LogStatusEnum.FAILED);
            update.setErrorMessage(e.getMessage());
            evalLogMapper.updateById(update);
        }
    }

    private String buildReportHtml(SysPredLog predLog, String algorithmName) {
        return """
                <!DOCTYPE html>
                <html lang="zh-CN">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>去雾效果对比报告</title>
                    <style>
                        * { margin: 0; padding: 0; box-sizing: border-box; }
                        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; padding: 20px; }
                        .container { max-width: 1200px; margin: 0 auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.1); overflow: hidden; }
                        .header { background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%); color: #fff; padding: 30px; }
                        .header h1 { font-size: 24px; margin-bottom: 8px; }
                        .header .meta { font-size: 14px; opacity: 0.85; }
                        .section { padding: 24px 30px; border-bottom: 1px solid #eee; }
                        .section:last-child { border-bottom: none; }
                        .section h2 { font-size: 18px; color: #667eea; margin-bottom: 16px; }
                        .comparison { display: flex; gap: 20px; flex-wrap: wrap; }
                        .image-card { flex: 1; min-width: 280px; }
                        .image-card .label { font-size: 14px; color: #666; margin-bottom: 8px; font-weight: 500; }
                        .image-card img { width: 100%%; border-radius: 6px; border: 1px solid #e0e0e0; }
                        .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
                        .info-item { background: #f8f9ff; padding: 12px 16px; border-radius: 6px; }
                        .info-item .label { font-size: 12px; color: #999; margin-bottom: 4px; }
                        .info-item .value { font-size: 16px; font-weight: 500; }
                        .footer { text-align: center; padding: 20px; color: #999; font-size: 12px; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>去雾效果对比报告</h1>
                            <div class="meta">算法：%s | 生成时间：%s</div>
                        </div>
                        <div class="section">
                            <h2>图片对比</h2>
                            <div class="comparison">
                                <div class="image-card">
                                    <div class="label">原图</div>
                                    <img src="%s" alt="原图" onerror="this.style.display='none'" />
                                </div>
                                <div class="image-card">
                                    <div class="label">处理结果</div>
                                    <img src="%s" alt="处理结果" onerror="this.style.display='none'" />
                                </div>
                            </div>
                        </div>
                        <div class="section">
                            <h2>处理信息</h2>
                            <div class="info-grid">
                                <div class="info-item">
                                    <div class="label">算法名称</div>
                                    <div class="value">%s</div>
                                </div>
                                <div class="info-item">
                                    <div class="label">算法ID</div>
                                    <div class="value">%d</div>
                                </div>
                                <div class="info-item">
                                    <div class="label">处理时间</div>
                                    <div class="value">%d ms</div>
                                </div>
                                <div class="info-item">
                                    <div class="label">任务状态</div>
                                    <div class="value">已完成</div>
                                </div>
                            </div>
                        </div>
                        <div class="footer">
                            本报告由 Dehaze 系统自动生成
                        </div>
                    </div>
                </body>
                </html>
                """
                .formatted(
                        algorithmName, LocalDateTime.now().format(DT_FMT),
                        predLog.getOriginUrl() != null ? predLog.getOriginUrl() : "",
                        predLog.getPredUrl() != null ? predLog.getPredUrl() : "",
                        algorithmName,
                        predLog.getAlgorithmId() != null ? predLog.getAlgorithmId() : 0,
                        predLog.getTime() != null ? predLog.getTime() : 0
                );
    }
}
