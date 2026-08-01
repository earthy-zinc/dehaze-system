package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AnalyzeForm;
import com.pei.dehaze.model.form.RecommendationFeedbackForm;
import com.pei.dehaze.model.form.RecommendationRuleForm;
import com.pei.dehaze.model.vo.IdVO;
import com.pei.dehaze.model.vo.ImageFeatureAnalysisVO;
import com.pei.dehaze.model.vo.RecommendationReportVO;
import com.pei.dehaze.model.vo.RecommendationRuleVO;
import com.pei.dehaze.model.vo.RecommendedAlgorithmVO;
import com.pei.dehaze.service.RecommendationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Tag(name = "16.推荐管理")
@RestController
@RequestMapping("/api/v1/recommendations")
@RequiredArgsConstructor
public class RecommendationController {

    private final RecommendationService recommendationService;

    // ============ 推荐查询接口 ============

    @Operation(summary = "图像特征分析（F-REC-001）")
    @PostMapping("/analyze")
    public Result<ImageFeatureAnalysisVO> analyze(@Valid @RequestBody AnalyzeForm form) {
        return Result.success(recommendationService.analyze(form));
    }

    @Operation(summary = "获取算法推荐（F-REC-002）")
    @GetMapping("/algorithms")
    public Result<List<RecommendedAlgorithmVO>> getAlgorithmRecommendations(
            @Parameter(description = "推荐记录ID") @RequestParam(required = false) Long analysisId,
            @Parameter(description = "图像MD5") @RequestParam(required = false) String imageMd5) {
        return Result.success(recommendationService.getAlgorithmRecommendations(analysisId, imageMd5));
    }

    // ============ 反馈接口 ============

    @Operation(summary = "提交推荐反馈（F-REC-003）")
    @PostMapping("/feedback")
    public Result<IdVO> submitFeedback(@Valid @RequestBody RecommendationFeedbackForm form) {
        return Result.success(recommendationService.submitFeedback(form));
    }

    // ============ 规则管理接口（管理员） ============

    @Operation(summary = "获取推荐规则配置")
    @GetMapping("/rules")
    @PreAuthorize("@ss.hasPerm('sys:recommendation:rule:view')")
    public Result<List<RecommendationRuleVO>> getRules() {
        return Result.success(recommendationService.getRules());
    }

    @Operation(summary = "更新推荐规则配置（新增/修改）")
    @PutMapping("/rules")
    @PreAuthorize("@ss.hasPerm('sys:recommendation:rule:edit')")
    public Result<Long> updateRule(
            @Parameter(description = "规则ID（0表示新增）") @RequestParam(defaultValue = "0") Long id,
            @Valid @RequestBody RecommendationRuleForm form) {
        return Result.success(recommendationService.updateRule(id, form));
    }

    // ============ 效果报表接口（管理员） ============

    @Operation(summary = "推荐效果报表")
    @GetMapping("/report")
    @PreAuthorize("@ss.hasPerm('sys:recommendation:report')")
    public Result<RecommendationReportVO> getReport(
            @Parameter(description = "开始日期(yyyy-MM-dd)") @RequestParam(required = false) String startDate,
            @Parameter(description = "结束日期(yyyy-MM-dd)") @RequestParam(required = false) String endDate) {
        return Result.success(recommendationService.getReport(startDate, endDate));
    }
}
