package com.pei.dehaze.controller;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.EvaluationForm;
import com.pei.dehaze.model.query.EvalLogQuery;
import com.pei.dehaze.model.vo.EvalLogVO;
import com.pei.dehaze.model.vo.EvaluationResultVO;
import com.pei.dehaze.service.SysEvalLogService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.*;

import java.util.LinkedHashMap;
import java.util.Map;

@Tag(name = "11.效果评估接口")
@RestController
@RequestMapping("/api/v1/evaluation")
@RequiredArgsConstructor
public class EvaluationController {

    private final SysEvalLogService evalLogService;

    @Operation(summary = "执行效果评估（PSNR/SSIM/LPIPS等，异步）")
    @PostMapping
    public Result<EvaluationResultVO> evaluate(@Valid @RequestBody EvaluationForm form) {
        EvaluationResultVO result = evalLogService.evaluate(form);
        return Result.success(result);
    }

    @Operation(summary = "查询评估任务状态")
    @GetMapping("/{taskId}")
    public Result<EvaluationResultVO> getTaskStatus(
            @Parameter(description = "评估任务ID") @PathVariable Long taskId) {
        var evalLog = evalLogService.getById(taskId);
        if (evalLog == null) {
            return Result.failed("评估任务不存在");
        }
        EvaluationResultVO result = new EvaluationResultVO();
        result.setLogId(evalLog.getId());
        result.setStatus(evalLog.getStatus());
        if (LogStatusEnum.COMPLETED == evalLog.getStatus()) {
            result.setMetrics(parseMetrics(evalLog.getResult()));
            result.setTime(evalLog.getTime());
        } else if (LogStatusEnum.FAILED == evalLog.getStatus()) {
            result.setErrorMessage(evalLog.getErrorMessage());
            result.setTime(evalLog.getTime());
        }
        return Result.success(result);
    }

    @Operation(summary = "获取评估日志列表")
    @GetMapping("/logs")
    public PageResult<EvalLogVO> getEvalLogs(@ParameterObject EvalLogQuery query) {
        Page<EvalLogVO> page = evalLogService.getEvalLogPage(query);
        return PageResult.success(page);
    }

    private Map<String, Double> parseMetrics(String resultJson) {
        if (resultJson == null || resultJson.isBlank()) {
            return null;
        }
        JSONObject json = JSONUtil.parseObj(resultJson);
        JSONObject metricsJson = json.getJSONObject("metrics");
        if (metricsJson == null) {
            return null;
        }
        Map<String, Double> metrics = new LinkedHashMap<>();
        for (String key : metricsJson.keySet()) {
            metrics.put(key, metricsJson.getDouble(key));
        }
        return metrics;
    }
}
