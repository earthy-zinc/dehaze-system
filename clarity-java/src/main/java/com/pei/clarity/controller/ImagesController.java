package com.pei.clarity.controller;

import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.clarity.common.enums.LogStatusEnum;
import com.pei.clarity.common.exception.BusinessException;
import com.pei.clarity.common.result.PageResult;
import com.pei.clarity.common.result.Result;
import com.pei.clarity.common.result.ResultCode;
import com.pei.clarity.model.entity.SysPredLog;
import com.pei.clarity.model.form.PredictionForm;
import com.pei.clarity.model.query.PredLogQuery;
import com.pei.clarity.model.vo.PredLogVO;
import com.pei.clarity.model.vo.PredictionResultVO;
import com.pei.clarity.security.util.SecurityUtils;
import com.pei.clarity.service.SysPredLogService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.*;

/**
 * 图像任务 API —— 提交-轮询任务模式（设计 §2.7）。
 *
 * <p>POST /api/v1/images/edits、GET /api/v1/images/tasks/{taskId}、
 * POST /api/v1/images/tasks/{taskId}/cancel、GET /api/v1/images/tasks。
 * 原 /prediction/batch 与 /prediction/quota 已退役。
 */
@Tag(name = "18.图像任务")
@RestController
@RequestMapping("/api/v1/images")
@RequiredArgsConstructor
public class ImagesController {

    private final SysPredLogService predLogService;

    @Operation(summary = "提交图像编辑任务（去雾处理，异步）")
    @PostMapping("/edits")
    public Result<PredictionResultVO> submitEdit(@Valid @RequestBody PredictionForm form) {
        return Result.success(predLogService.predict(form));
    }

    @Operation(summary = "查询图像任务状态")
    @GetMapping("/tasks/{taskId}")
    public Result<PredictionResultVO> getTaskStatus(@Parameter(description = "图像任务ID") @PathVariable Long taskId) {
        SysPredLog predLog = predLogService.getById(taskId);
        // 归属校验：仅任务本人可查询（含结果图 URL），他人任务与不存在任务同口径防枚举
        if (predLog == null || !SecurityUtils.getUserId().equals(predLog.getCreateBy())) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "图像任务不存在");
        }
        PredictionResultVO result = new PredictionResultVO();
        result.setTaskId(predLog.getId());
        result.setStatus(predLog.getStatus().getValue());
        if (LogStatusEnum.COMPLETED == predLog.getStatus()) {
            result.setResultUrl(predLog.getPredUrl());
            result.setResultMd5(predLog.getPredMd5());
            result.setOriginMd5(predLog.getOriginMd5());
            result.setTime(predLog.getTime());
        } else if (LogStatusEnum.FAILED == predLog.getStatus()) {
            result.setErrorMessage(predLog.getErrorMessage());
            result.setTime(predLog.getTime());
        }
        return Result.success(result);
    }

    @Operation(summary = "取消图像任务（幂等：仅处理中任务可取消，其余返回当前状态）")
    @PostMapping("/tasks/{taskId}/cancel")
    public Result<PredictionResultVO> cancelTask(@Parameter(description = "图像任务ID") @PathVariable Long taskId) {
        SysPredLog predLog = predLogService.getById(taskId);
        if (predLog == null || !SecurityUtils.getUserId().equals(predLog.getCreateBy())) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "图像任务不存在");
        }
        if (LogStatusEnum.PROCESSING == predLog.getStatus()) {
            // 状态流转带 processing 前置条件：并发取消只有一个生效
            boolean ok = predLogService.update(new LambdaUpdateWrapper<SysPredLog>()
                    .eq(SysPredLog::getId, taskId)
                    .eq(SysPredLog::getStatus, LogStatusEnum.PROCESSING)
                    .set(SysPredLog::getStatus, LogStatusEnum.CANCELLED)
                    .set(SysPredLog::getErrorMessage, "任务已取消"));
            if (ok) {
                predLog.setStatus(LogStatusEnum.CANCELLED);
            }
        }
        PredictionResultVO result = new PredictionResultVO();
        result.setTaskId(predLog.getId());
        result.setStatus(predLog.getStatus().getValue());
        return Result.success(result);
    }

    @Operation(summary = "图像任务历史列表（当前用户，分页）")
    @GetMapping("/tasks")
    public PageResult<PredLogVO> listTasks(@Valid @ParameterObject PredLogQuery query) {
        Page<PredLogVO> page = predLogService.getPredLogPage(query);
        return PageResult.success(page);
    }
}
