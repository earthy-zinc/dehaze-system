package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.query.AiObservabilityCostsQuery;
import com.pei.dehaze.model.query.AiObservabilityTraceQuery;
import com.pei.dehaze.model.query.AiObservabilityTimelineQuery;
import com.pei.dehaze.model.query.AiObservabilityTrendsQuery;
import com.pei.dehaze.model.vo.AiObservabilityCostsVO;
import com.pei.dehaze.model.vo.AiObservabilitySummaryVO;
import com.pei.dehaze.model.vo.AiObservabilityTimelineVO;
import com.pei.dehaze.model.vo.AiObservabilityTraceDetailVO;
import com.pei.dehaze.model.vo.AiObservabilityTraceItemVO;
import com.pei.dehaze.model.vo.AiObservabilityTrendVO;
import com.pei.dehaze.service.AiObservabilityService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * AI 可观测性（F-M08-013）：异常总览、过程链检索/详情/导出、会话审计时间线、资源消耗与性能趋势。
 *
 * <p>除过程链详情与会话时间线（登录用户可查，普通用户仅自己会话）外，其余端点需
 * {@code ai:conversation:audit} 权限。
 *
 * @author dehaze
 */
@Tag(name = "30.AI可观测性")
@RestController
@RequestMapping("/api/v1/ai")
@RequiredArgsConstructor
public class AiObservabilityController {

    private final AiObservabilityService observabilityService;

    @Operation(summary = "异常总览统计")
    @GetMapping("/observability/summary")
    @PreAuthorize("@ss.hasPerm('ai:conversation:audit')")
    public Result<AiObservabilitySummaryVO> summary() {
        return Result.success(observabilityService.summary());
    }

    @Operation(summary = "过程链检索")
    @GetMapping("/observability/traces")
    @PreAuthorize("@ss.hasPerm('ai:conversation:audit')")
    public PageResult<AiObservabilityTraceItemVO> traces(@Valid @ParameterObject AiObservabilityTraceQuery query) {
        IPage<AiObservabilityTraceItemVO> page = observabilityService.listTraces(query);
        return PageResult.success(page);
    }

    @Operation(summary = "过程链导出(CSV)")
    @GetMapping("/observability/traces/export")
    @PreAuthorize("@ss.hasPerm('ai:conversation:audit')")
    public ResponseEntity<byte[]> exportTraces(@Valid @ParameterObject AiObservabilityTraceQuery query) {
        byte[] payload = observabilityService.exportTraces(query);
        return ResponseEntity.ok()
                .contentType(new MediaType("text", "csv", StandardCharsets.UTF_8))
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"ai_traces.csv\"")
                .body(payload);
    }

    @Operation(summary = "过程链详情")
    @GetMapping("/observability/traces/{traceId}")
    public Result<AiObservabilityTraceDetailVO> traceDetail(
            @Parameter(description = "过程链ID") @PathVariable String traceId) {
        return Result.success(observabilityService.getTrace(traceId));
    }

    @Operation(summary = "会话审计时间线")
    @GetMapping("/observability/conversations/{conversationId}/timeline")
    public Result<AiObservabilityTimelineVO> timeline(
            @Parameter(description = "会话ID") @PathVariable Long conversationId,
            @Valid @ParameterObject AiObservabilityTimelineQuery query) {
        return Result.success(observabilityService.getTimeline(conversationId, query.getInclude()));
    }

    @Operation(summary = "会话时间线导出(JSON全量含raw报文)")
    @GetMapping("/observability/conversations/{conversationId}/timeline/export")
    @PreAuthorize("@ss.hasPerm('ai:conversation:audit')")
    public ResponseEntity<byte[]> exportTimeline(
            @Parameter(description = "会话ID") @PathVariable Long conversationId) {
        byte[] payload = observabilityService.exportTimelineJson(conversationId);
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "attachment; filename=\"conversation_" + conversationId + "_timeline.json\"")
                .body(payload);
    }

    @Operation(summary = "资源消耗聚合")
    @GetMapping("/observability/costs")
    @PreAuthorize("@ss.hasPerm('ai:conversation:audit')")
    public Result<AiObservabilityCostsVO> costs(@Valid @ParameterObject AiObservabilityCostsQuery query) {
        return Result.success(observabilityService.costs(query));
    }

    @Operation(summary = "性能趋势")
    @GetMapping("/observability/trends")
    @PreAuthorize("@ss.hasPerm('ai:conversation:audit')")
    public Result<List<AiObservabilityTrendVO>> trends(@Valid @ParameterObject AiObservabilityTrendsQuery query) {
        return Result.success(observabilityService.trends(query));
    }
}
