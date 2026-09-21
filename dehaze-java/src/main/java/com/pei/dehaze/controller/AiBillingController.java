package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AiBillingAdjustForm;
import com.pei.dehaze.model.form.AiModelCostForm;
import com.pei.dehaze.model.form.AiModelCostUpdateForm;
import com.pei.dehaze.model.form.AiReconcileImportForm;
import com.pei.dehaze.model.form.AiRefundAuditForm;
import com.pei.dehaze.model.form.AiRefundCreateForm;
import com.pei.dehaze.model.query.AiBalanceQuery;
import com.pei.dehaze.model.query.AiBillingAnomalyQuery;
import com.pei.dehaze.model.query.AiBillingRecordQuery;
import com.pei.dehaze.model.query.AiBillingStatQuery;
import com.pei.dehaze.model.query.AiCreditLogQuery;
import com.pei.dehaze.model.query.AiModelCostQuery;
import com.pei.dehaze.model.query.AiRefundQuery;
import com.pei.dehaze.model.vo.AiBalanceVO;
import com.pei.dehaze.model.vo.AiBillVO;
import com.pei.dehaze.model.vo.AiBillingAnomalyVO;
import com.pei.dehaze.model.vo.AiBillingRecordVO;
import com.pei.dehaze.model.vo.AiBillingStatVO;
import com.pei.dehaze.model.vo.AiBillingSummaryVO;
import com.pei.dehaze.model.vo.AiCostStatVO;
import com.pei.dehaze.model.vo.AiCreditLogVO;
import com.pei.dehaze.model.vo.AiModelCostVO;
import com.pei.dehaze.model.vo.AiRefundVO;
import com.pei.dehaze.service.AiBillingService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

/**
 * AI 计费管理（用户端余额/明细/账单/退款；管理端统计/调整/审核/成本/对账）。
 *
 * @author dehaze
 */
@Tag(name = "31.AI计费管理")
@RestController
@RequestMapping("/api/v1/ai-billing")
@RequiredArgsConstructor
public class AiBillingController {

    private final AiBillingService billingService;

    // ── 用户端 ──────────────────────────────────────────────

    @Operation(summary = "用户余额查询")
    @GetMapping("/balance")
    public Result<AiBalanceVO> balance(@Valid @ParameterObject AiBalanceQuery query) {
        return Result.success(billingService.getBalance(query.getUserId()));
    }

    @Operation(summary = "消耗汇总查询")
    @GetMapping("/summary")
    public Result<AiBillingSummaryVO> summary(
            @Parameter(description = "统计时段：day-当日 / month-当月")
            @RequestParam(defaultValue = "day") String dimension) {
        return Result.success(billingService.getSummary(dimension));
    }

    @Operation(summary = "计费明细查询")
    @GetMapping("/records")
    public PageResult<AiBillingRecordVO> records(@Valid @ParameterObject AiBillingRecordQuery query) {
        IPage<AiBillingRecordVO> page = billingService.listRecords(query);
        return PageResult.success(page);
    }

    @Operation(summary = "余额流水查询")
    @GetMapping("/credit-logs")
    public PageResult<AiCreditLogVO> creditLogs(@Valid @ParameterObject AiCreditLogQuery query) {
        IPage<AiCreditLogVO> page = billingService.listCreditLogs(query);
        return PageResult.success(page);
    }

    @Operation(summary = "月结账单查询")
    @GetMapping("/bills/{month}")
    public Result<AiBillVO> bill(@Parameter(description = "账期月份（YYYY-MM）") @PathVariable String month) {
        return Result.success(billingService.getBill(month));
    }

    @Operation(summary = "账单下载")
    @GetMapping("/bills/{month}/download")
    public Result<AiBillVO> downloadBill(
            @Parameter(description = "账期月份（YYYY-MM）") @PathVariable String month) {
        return Result.success(billingService.getBill(month));
    }

    @Operation(summary = "退款申请")
    @PostMapping("/refunds")
    public Result<AiRefundVO> applyRefund(@Valid @RequestBody AiRefundCreateForm form) {
        return Result.success(billingService.applyRefund(form));
    }

    // ── 管理端 ──────────────────────────────────────────────

    @Operation(summary = "退款申请列表")
    @GetMapping("/refunds")
    @PreAuthorize("@ss.hasPerm('ai:billing:refund')")
    public PageResult<AiRefundVO> refunds(@Valid @ParameterObject AiRefundQuery query) {
        IPage<AiRefundVO> page = billingService.listRefunds(query);
        return PageResult.success(page);
    }

    @Operation(summary = "管理员计费统计")
    @GetMapping("/stats")
    @PreAuthorize("@ss.hasPerm('ai:billing:stat')")
    public Result<List<AiBillingStatVO>> stats(@Valid @ParameterObject AiBillingStatQuery query) {
        return Result.success(billingService.getStats(query));
    }

    @Operation(summary = "管理员手动调整积分")
    @PostMapping("/adjust")
    @PreAuthorize("@ss.hasPerm('ai:billing:adjust')")
    public Result<AiBalanceVO> adjust(@Valid @RequestBody AiBillingAdjustForm form) {
        return Result.success(billingService.adjustCredits(form));
    }

    @Operation(summary = "退款审核")
    @PostMapping("/refunds/{refundId}/audit")
    @PreAuthorize("@ss.hasPerm('ai:billing:refund')")
    public Result<AiRefundVO> auditRefund(@PathVariable Long refundId,
                                          @Valid @RequestBody AiRefundAuditForm form) {
        return Result.success(billingService.auditRefund(refundId, form));
    }

    @Operation(summary = "异常计费记录查询")
    @GetMapping("/anomalies")
    @PreAuthorize("@ss.hasPerm('ai:billing:stat')")
    public PageResult<AiBillingAnomalyVO> anomalies(@Valid @ParameterObject AiBillingAnomalyQuery query) {
        IPage<AiBillingAnomalyVO> page = billingService.listAnomalies(query);
        return PageResult.success(page);
    }

    @Operation(summary = "成本单价列表")
    @GetMapping("/costs")
    @PreAuthorize("@ss.hasPerm('ai:billing:cost')")
    public PageResult<AiModelCostVO> costs(@Valid @ParameterObject AiModelCostQuery query) {
        IPage<AiModelCostVO> page = billingService.listCosts(query);
        return PageResult.success(page);
    }

    @Operation(summary = "新增成本单价")
    @PostMapping("/costs")
    @PreAuthorize("@ss.hasPerm('ai:billing:cost')")
    public Result<AiModelCostVO> createCost(@Valid @RequestBody AiModelCostForm form) {
        return Result.success(billingService.createCost(form));
    }

    @Operation(summary = "更新成本单价")
    @PutMapping("/costs/{costId}")
    @PreAuthorize("@ss.hasPerm('ai:billing:cost')")
    public Result<AiModelCostVO> updateCost(@PathVariable Long costId,
                                           @Valid @RequestBody AiModelCostUpdateForm form) {
        return Result.success(billingService.updateCost(costId, form));
    }

    @Operation(summary = "删除成本单价")
    @DeleteMapping("/costs/{costId}")
    @PreAuthorize("@ss.hasPerm('ai:billing:cost')")
    public Result<Void> deleteCost(@PathVariable Long costId) {
        billingService.deleteCost(costId);
        return Result.success();
    }

    @Operation(summary = "成本-利润统计")
    @GetMapping("/cost-stats")
    @PreAuthorize("@ss.hasPerm('ai:billing:cost')")
    public Result<List<AiCostStatVO>> costStats(
            @Parameter(description = "开始时间") @RequestParam(required = false) String startTime,
            @Parameter(description = "结束时间") @RequestParam(required = false) String endTime,
            @Parameter(description = "分组维度：overall/model/provider")
            @RequestParam(defaultValue = "overall") String groupBy,
            @RequestParam(required = false) String modelId,
            @RequestParam(required = false) Long providerId) {
        return Result.success(billingService.getCostStats(startTime, endTime, groupBy, modelId, providerId));
    }

    @Operation(summary = "供应商账单导入")
    @PostMapping("/reconcile/import")
    @PreAuthorize("@ss.hasPerm('ai:billing:cost')")
    public Result<Map<String, Object>> importReconcile(@Valid @RequestBody AiReconcileImportForm form) {
        return Result.success(billingService.importReconcile(form));
    }
}
