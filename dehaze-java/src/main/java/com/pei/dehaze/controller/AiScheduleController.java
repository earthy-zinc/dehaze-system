package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AiScheduleCreateForm;
import com.pei.dehaze.model.form.AiScheduleStatusForm;
import com.pei.dehaze.model.form.AiScheduleUpdateForm;
import com.pei.dehaze.model.query.AiSchedulePageQuery;
import com.pei.dehaze.model.query.PageParamQuery;
import com.pei.dehaze.model.vo.AiNextTimesVO;
import com.pei.dehaze.model.vo.AiScheduleHistoryVO;
import com.pei.dehaze.model.vo.AiScheduleVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiScheduleService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * AI 定时调度（手动触发执行 /run 属转发域）。
 *
 * <p>{@code @Validated} 开启方法级参数校验：{@code count} 过去无上下界，
 * 越界会直达 Cron 预览计算，与 python {@code Query(5, ge=1, le=20)} 分叉。
 *
 * @author dehaze
 */
@Tag(name = "29.AI对话")
@RestController
@RequestMapping("/api/v1/ai/scheduled-tasks")
@RequiredArgsConstructor
@Validated
public class AiScheduleController {

    private final AiScheduleService scheduleService;

    @Operation(summary = "创建定时任务")
    @PostMapping
    public Result<AiScheduleVO> create(@Valid @RequestBody AiScheduleCreateForm form) {
        return Result.success(scheduleService.create(SecurityUtils.getUserId(), form));
    }

    @Operation(summary = "定时任务列表")
    @GetMapping
    public PageResult<AiScheduleVO> list(@Valid @ParameterObject AiSchedulePageQuery query) {
        return PageResult.success(scheduleService.list(SecurityUtils.getUserId(), query));
    }

    @Operation(summary = "Cron 解释与下次执行时间预览")
    @GetMapping("/next-times")
    public Result<AiNextTimesVO> previewNextTimes(
            @Parameter(description = "Cron 触发规则(5位表达式或常用频率标识)") @RequestParam String cron,
            @Parameter(description = "返回的触发时间次数(1~20)") @Min(1) @Max(20)
            @RequestParam(defaultValue = "5") int count) {
        return Result.success(scheduleService.previewNextTimes(cron, count));
    }

    @Operation(summary = "定时任务详情")
    @GetMapping("/{scheduleId}")
    public Result<AiScheduleVO> detail(@Parameter(description = "任务ID") @PathVariable Long scheduleId) {
        return Result.success(scheduleService.getDetail(SecurityUtils.getUserId(), scheduleId));
    }

    @Operation(summary = "更新定时任务")
    @PutMapping("/{scheduleId}")
    public Result<AiScheduleVO> update(@Parameter(description = "任务ID") @PathVariable Long scheduleId,
                                       @Valid @RequestBody AiScheduleUpdateForm form) {
        return Result.success(scheduleService.update(SecurityUtils.getUserId(), scheduleId, form));
    }

    @Operation(summary = "启停定时任务")
    @PatchMapping("/{scheduleId}/status")
    public Result<Void> setStatus(@Parameter(description = "任务ID") @PathVariable Long scheduleId,
                                  @Valid @RequestBody AiScheduleStatusForm form) {
        scheduleService.setEnabled(SecurityUtils.getUserId(), scheduleId, form.getEnabled());
        return Result.success();
    }

    @Operation(summary = "删除定时任务")
    @DeleteMapping("/{scheduleId}")
    public Result<Void> delete(@Parameter(description = "任务ID") @PathVariable Long scheduleId) {
        scheduleService.delete(SecurityUtils.getUserId(), scheduleId);
        return Result.success();
    }

    @Operation(summary = "执行历史")
    @GetMapping("/{scheduleId}/history")
    public PageResult<AiScheduleHistoryVO> history(
            @Parameter(description = "任务ID") @PathVariable Long scheduleId,
            @Valid @ParameterObject PageParamQuery query) {
        return PageResult.success(scheduleService.listHistory(SecurityUtils.getUserId(), scheduleId,
                query.getPageNum(), query.getPageSize()));
    }
}
