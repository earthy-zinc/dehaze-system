package com.pei.dehaze.controller;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.model.entity.SysEvalLog;
import com.pei.dehaze.model.form.EvaluationForm;
import com.pei.dehaze.model.query.EvalLogQuery;
import com.pei.dehaze.model.vo.EvalLogVO;
import com.pei.dehaze.model.vo.EvalMetricsVO;
import com.pei.dehaze.model.vo.EvaluationResultVO;
import com.pei.dehaze.security.util.SecurityUtils;
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
import java.util.stream.Collectors;

@Tag(name = "11.效果评估接口")
@RestController
@RequestMapping("/api/v1/evaluation")
@RequiredArgsConstructor
public class EvaluationController {

    private final SysEvalLogService evalLogService;
    private final SysEvalLogMapper evalLogMapper;

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

    @Operation(summary = "获取评估指标历史（当前用户）")
    @GetMapping("/metrics")
    public PageResult<EvalMetricsVO> getMetrics(@ParameterObject EvalLogQuery query) {
        Long userId = SecurityUtils.getUserId();
        Page<SysEvalLog> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysEvalLog> wrapper = new LambdaQueryWrapper<SysEvalLog>()
                .eq(SysEvalLog::getCreateBy, userId)
                .eq(query.getAlgorithmId() != null, SysEvalLog::getAlgorithmId, query.getAlgorithmId())
                .eq(SysEvalLog::getStatus, LogStatusEnum.COMPLETED)
                .orderByDesc(SysEvalLog::getCreateTime);
        Page<SysEvalLog> result = evalLogMapper.selectPage(page, wrapper);
        Page<EvalMetricsVO> voPage = new Page<>(result.getCurrent(), result.getSize(), result.getTotal());
        voPage.setRecords(result.getRecords().stream().map(log -> {
            EvalMetricsVO vo = new EvalMetricsVO();
            vo.setId(log.getId());
            vo.setAlgorithmId(log.getAlgorithmId());
            vo.setPredUrl(log.getPredUrl());
            vo.setGtUrl(log.getGtUrl());
            vo.setTime(log.getTime());
            vo.setStatus(log.getStatus());
            vo.setErrorMessage(log.getErrorMessage());
            vo.setCreateTime(log.getCreateTime());
            vo.setMetrics(parseMetrics(log.getResult()));
            return vo;
        }).collect(Collectors.toList()));
        return PageResult.success(voPage);
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
        Map<String, Double> metrics = new LinkedHashMap<>();
        for (String key : json.keySet()) {
            metrics.put(key, json.getDouble(key));
        }
        return metrics;
    }
}
