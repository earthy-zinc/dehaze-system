package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.vo.AiUsageStatsVO;
import com.pei.dehaze.service.AiUsageStatsService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDateTime;

/**
 * AI 运营统计（供应商健康看板/模型用量分布/降级与故障）。
 *
 * @author dehaze
 */
@Tag(name = "29.AI模型管理")
@RestController
@RequestMapping("/api/v1/ai")
@RequiredArgsConstructor
public class AiUsageStatsController {

    private final AiUsageStatsService usageStatsService;

    @Operation(summary = "运营统计(供应商健康看板/模型用量分布/降级与故障)")
    @GetMapping("/usage/stats")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<AiUsageStatsVO> stats(
            @Parameter(description = "起始时间") @RequestParam(required = false)
            @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime startTime,
            @Parameter(description = "结束时间") @RequestParam(required = false)
            @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime endTime) {
        return Result.success(usageStatsService.getUsageStats(startTime, endTime));
    }
}
