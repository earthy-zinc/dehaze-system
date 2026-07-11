package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.PredictionForm;
import com.pei.dehaze.model.query.PredLogQuery;
import com.pei.dehaze.model.vo.PredLogVO;
import com.pei.dehaze.model.vo.PredictionResultVO;
import com.pei.dehaze.service.SysPredLogService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.*;

/**
 * 模型预测控制器 —— 去雾处理核心API入口
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Tag(name = "10.模型预测接口")
@RestController
@RequestMapping("/api/v1/prediction")
@RequiredArgsConstructor
public class PredictionController {

    private final SysPredLogService predLogService;

    @Operation(summary = "执行模型预测（去雾处理）")
    @PostMapping
    public Result<PredictionResultVO> predict(@Valid @RequestBody PredictionForm form) {
        PredictionResultVO result = predLogService.predict(form);
        return Result.success(result);
    }

    @Operation(summary = "查询预测任务状态")
    @GetMapping("/{taskId}")
    public Result<PredictionResultVO> getTaskStatus(
            @Parameter(description = "预测任务ID") @PathVariable Long taskId) {
        // 查询预测日志判断状态
        var predLog = predLogService.getById(taskId);
        if (predLog == null) {
            return Result.failed("预测任务不存在");
        }
        PredictionResultVO result = new PredictionResultVO();
        result.setLogId(predLog.getId());
        result.setResultUrl(predLog.getPredUrl());
        result.setTime(predLog.getTime());
        return Result.success(result);
    }

    @Operation(summary = "获取预测日志列表")
    @GetMapping("/logs")
    public PageResult<PredLogVO> getPredLogs(@ParameterObject PredLogQuery query) {
        Page<PredLogVO> page = predLogService.getPredLogPage(query);
        return PageResult.success(page);
    }
}
