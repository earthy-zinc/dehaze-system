package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AiEvalReviewSubmitForm;
import com.pei.dehaze.model.vo.AiEvalCompareVO;
import com.pei.dehaze.model.vo.AiEvalOverviewVO;
import com.pei.dehaze.model.vo.AiEvalReviewDetailVO;
import com.pei.dehaze.model.vo.AiEvalReviewQueueVO;
import com.pei.dehaze.model.vo.AiEvalTrendVO;
import com.pei.dehaze.model.vo.AiJudgeStatusVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiEvalCenterService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.RequiredArgsConstructor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 评测中心（跨 Agent 聚合：总览/趋势/对比/判分状态/人工复核）。
 *
 * <p>{@code @Validated} 开启方法级参数校验：非分页的标量查询参数（limit/status）
 * 过去只靠 Spring 类型转换兜住格式错误，越界值（如 limit=0 或 501）会静默透传，
 * 与 python 侧 {@code Query(ge=, le=)} 的行为分叉；收敛到注解校验后可统一返回 400 + A0400。
 *
 * @author dehaze
 */
@Tag(name = "29.AI对话")
@RestController
@RequestMapping("/api/v1/ai/eval-center")
@RequiredArgsConstructor
@Validated
public class AiEvalCenterController {

    private final AiEvalCenterService evalCenterService;

    @Operation(summary = "评测总览（各 Agent 最近得分/门禁状态/退化标识）")
    @GetMapping("/overview")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<List<AiEvalOverviewVO>> overview() {
        return Result.success(evalCenterService.overview());
    }

    @Operation(summary = "评测历史趋势")
    @GetMapping("/trends")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<List<AiEvalTrendVO>> trends(
            @Parameter(description = "Agent ID") @RequestParam(required = false) Long agentId,
            @Parameter(description = "起始时间") @RequestParam(required = false)
            @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime startTime,
            @Parameter(description = "结束时间") @RequestParam(required = false)
            @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime endTime,
            @Parameter(description = "条数上限(1~500)") @Min(1) @Max(500)
            @RequestParam(defaultValue = "100") int limit) {
        return Result.success(evalCenterService.trends(agentId, startTime, endTime, limit));
    }

    @Operation(summary = "两次评测 run 得分对比")
    @GetMapping("/runs/{runId}/compare")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiEvalCompareVO> compare(@Parameter(description = "本次评测记录ID") @PathVariable Long runId,
                                           @Parameter(description = "基准评测记录ID") @RequestParam Long baseRunId) {
        return Result.success(evalCenterService.compareRuns(runId, baseRunId));
    }

    @Operation(summary = "判分状态")
    @GetMapping("/judge-status")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiJudgeStatusVO> judgeStatus() {
        return Result.success(evalCenterService.judgeStatus());
    }

    @Operation(summary = "人工复核队列")
    @GetMapping("/reviews")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiEvalReviewQueueVO> reviews(
            @Parameter(description = "复核状态过滤(1:待复核;2:已复核)") @Min(1) @Max(2)
            @RequestParam(required = false) Integer status) {
        return Result.success(evalCenterService.listReviews(status));
    }

    @Operation(summary = "复核详情（样本输入/期望 + 实际输出 + 四维得分）")
    @GetMapping("/runs/{runId}/samples/{sampleId}")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiEvalReviewDetailVO> reviewDetail(
            @Parameter(description = "评测记录ID") @PathVariable Long runId,
            @Parameter(description = "样本ID") @PathVariable Long sampleId) {
        return Result.success(evalCenterService.reviewDetail(runId, sampleId));
    }

    @Operation(summary = "复核结果回填（判定一致/不一致 + 备注）")
    @PostMapping("/reviews/{reviewId}")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<Map<String, Object>> submitReview(
            @Parameter(description = "复核记录ID") @PathVariable Long reviewId,
            @Valid @RequestBody AiEvalReviewSubmitForm form) {
        return Result.success(evalCenterService.submitReview(reviewId, form.getAgree(), form.getRemark(),
                SecurityUtils.getUserId()));
    }
}
