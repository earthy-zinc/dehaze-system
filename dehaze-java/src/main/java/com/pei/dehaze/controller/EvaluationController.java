package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
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

/**
 * 效果评估控制器 —— 评估去雾处理效果
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Tag(name = "11.效果评估接口")
@RestController
@RequestMapping("/api/v1/evaluation")
@RequiredArgsConstructor
public class EvaluationController {

    private final SysEvalLogService evalLogService;

    @Operation(summary = "执行效果评估（PSNR/SSIM/LPIPS等）")
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
        result.setTime(evalLog.getTime());
        return Result.success(result);
    }

    @Operation(summary = "获取评估日志列表")
    @GetMapping("/logs")
    public PageResult<EvalLogVO> getEvalLogs(@ParameterObject EvalLogQuery query) {
        Page<EvalLogVO> page = evalLogService.getEvalLogPage(query);
        return PageResult.success(page);
    }
}
