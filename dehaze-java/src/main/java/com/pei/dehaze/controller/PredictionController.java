package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.BatchPredictionForm;
import com.pei.dehaze.model.form.PredictionForm;
import com.pei.dehaze.model.query.PredLogQuery;
import com.pei.dehaze.model.vo.BatchPredictionResultVO;
import com.pei.dehaze.model.vo.PredLogVO;
import com.pei.dehaze.model.vo.PredictionQuotaVO;
import com.pei.dehaze.model.vo.PredictionResultVO;
import com.pei.dehaze.service.SysPredLogService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.*;

@Tag(name = "18.去雾处理")
@RestController
@RequestMapping("/api/v1/prediction")
@RequiredArgsConstructor
public class PredictionController {

    private final SysPredLogService predLogService;

    @Operation(summary = "执行模型预测（去雾处理，异步）")
    @PostMapping
    public Result<PredictionResultVO> predict(@Valid @RequestBody PredictionForm form) {
        PredictionResultVO result = predLogService.predict(form);
        return Result.success(result);
    }

    @Operation(summary = "查询预测任务状态")
    @GetMapping("/{taskId}")
    public Result<PredictionResultVO> getTaskStatus(
            @Parameter(description = "预测任务ID") @PathVariable Long taskId) {
        var predLog = predLogService.getById(taskId);
        if (predLog == null) {
            return Result.failed("预测任务不存在");
        }
        PredictionResultVO result = new PredictionResultVO();
        result.setLogId(predLog.getId());
        result.setStatus(predLog.getStatus());
        if (LogStatusEnum.COMPLETED == predLog.getStatus()) {
            result.setResultUrl(predLog.getPredUrl());
            result.setTime(predLog.getTime());
        } else if (LogStatusEnum.FAILED == predLog.getStatus()) {
            result.setErrorMessage(predLog.getErrorMessage());
            result.setTime(predLog.getTime());
        }
        return Result.success(result);
    }

    @Operation(summary = "获取预测日志列表")
    @GetMapping("/logs")
    public PageResult<PredLogVO> getPredLogs(@ParameterObject PredLogQuery query) {
        Page<PredLogVO> page = predLogService.getPredLogPage(query);
        return PageResult.success(page);
    }

    @Operation(summary = "批量处理（一次提交多张图片）")
    @PostMapping("/batch")
    public Result<BatchPredictionResultVO> batchPredict(@Valid @RequestBody BatchPredictionForm form) {
        return Result.success(predLogService.batchPredict(form));
    }

    @Operation(summary = "查询用户剩余处理次数")
    @GetMapping("/quota")
    public Result<PredictionQuotaVO> getQuota() {
        return Result.success(predLogService.getQuota());
    }
}
