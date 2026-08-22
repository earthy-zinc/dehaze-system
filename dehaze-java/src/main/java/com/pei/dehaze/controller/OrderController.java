package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AutoRenewConfigForm;
import com.pei.dehaze.model.form.OrderCreateForm;
import com.pei.dehaze.model.form.PayRequest;
import com.pei.dehaze.model.form.RefundApplyForm;
import com.pei.dehaze.model.form.RefundAuditForm;
import com.pei.dehaze.model.query.MyOrderQuery;
import com.pei.dehaze.model.query.OrderPageQuery;
import com.pei.dehaze.model.query.RefundPageQuery;
import com.pei.dehaze.model.vo.AutoRenewConfigVO;
import com.pei.dehaze.model.vo.MyOrderVO;
import com.pei.dehaze.model.vo.OrderDetailVO;
import com.pei.dehaze.model.vo.OrderPageVO;
import com.pei.dehaze.model.vo.OrderStatsVO;
import com.pei.dehaze.model.vo.PayResult;
import com.pei.dehaze.model.vo.RefundRecordVO;
import com.pei.dehaze.service.OrderService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;

@Tag(name = "13.订单管理")
@RestController
@RequestMapping("/api/v1/orders")
@RequiredArgsConstructor
public class OrderController {

    private final OrderService orderService;

    @Operation(summary = "用户端：创建订单")
    @PostMapping
    public Result<PayResult> create(@Valid @RequestBody OrderCreateForm form) {
        return Result.success(orderService.create(form));
    }

    @Operation(summary = "用户端：我的订单列表")
    @GetMapping("/my")
    public PageResult<MyOrderVO> listMy(@ParameterObject MyOrderQuery query) {
        Page<MyOrderVO> page = orderService.listMy(query);
        return PageResult.success(page);
    }

    @Operation(summary = "用户端/后台：订单详情")
    @GetMapping("/{orderNo}")
    public Result<OrderDetailVO> getDetail(@Parameter(description = "订单号") @PathVariable String orderNo) {
        return Result.success(orderService.getDetail(orderNo));
    }

    @Operation(summary = "用户端：取消订单")
    @PutMapping("/{orderNo}/cancel")
    public Result<Void> cancel(@Parameter(description = "订单号") @PathVariable String orderNo,
                               @Parameter(description = "取消原因", required = true) @RequestParam String reason) {
        orderService.cancel(orderNo, reason);
        return Result.success();
    }

    @Operation(summary = "用户端：发起支付")
    @PostMapping("/{orderNo}/pay")
    public Result<PayResult> pay(@Parameter(description = "订单号") @PathVariable String orderNo,
                                 @Valid @RequestBody PayRequest request) {
        return Result.success(orderService.pay(orderNo, request));
    }

    @Operation(summary = "用户端：申请退款")
    @PostMapping("/{orderNo}/refund")
    public Result<Void> applyRefund(@Parameter(description = "订单号") @PathVariable String orderNo,
                                    @Valid @RequestBody RefundApplyForm form) {
        orderService.applyRefund(orderNo, form);
        return Result.success();
    }

    @Operation(summary = "用户端：修改自动续费设置")
    @PutMapping("/auto-renew/config")
    public Result<Void> updateAutoRenewConfig(@Valid @RequestBody AutoRenewConfigForm form) {
        orderService.updateAutoRenewConfig(form);
        return Result.success();
    }

    @Operation(summary = "用户端：查询自动续费配置")
    @GetMapping("/auto-renew/config")
    public Result<AutoRenewConfigVO> getAutoRenewConfig(
            @Parameter(description = "套餐ID", required = true) @RequestParam Long packageId) {
        return Result.success(orderService.getAutoRenewConfig(packageId));
    }

    @Operation(summary = "后台：订单分页列表")
    @GetMapping("/page")
    @PreAuthorize("@ss.hasPerm('order:list')")
    public PageResult<OrderPageVO> getPage(@ParameterObject OrderPageQuery query) {
        Page<OrderPageVO> page = orderService.getPage(query);
        return PageResult.success(page);
    }

    @Operation(summary = "后台：退款审核列表")
    @GetMapping("/refunds/page")
    @PreAuthorize("@ss.hasPerm('order:refund:list')")
    public PageResult<RefundRecordVO> listRefunds(@ParameterObject RefundPageQuery query) {
        Page<RefundRecordVO> page = orderService.listRefunds(query);
        return PageResult.success(page);
    }

    @Operation(summary = "后台：退款审核通过")
    @PutMapping("/refunds/{refundId}/approve")
    @PreAuthorize("@ss.hasPerm('order:refund:approve')")
    public Result<Void> approveRefund(@Parameter(description = "退款ID") @PathVariable Long refundId,
                                      @Valid @RequestBody RefundAuditForm form) {
        orderService.approveRefund(refundId, form);
        return Result.success();
    }

    @Operation(summary = "后台：退款审核驳回")
    @PutMapping("/refunds/{refundId}/reject")
    @PreAuthorize("@ss.hasPerm('order:refund:approve')")
    public Result<Void> rejectRefund(@Parameter(description = "退款ID") @PathVariable Long refundId,
                                     @Valid @RequestBody RefundAuditForm form) {
        orderService.rejectRefund(refundId, form);
        return Result.success();
    }

    @Operation(summary = "后台：订单统计")
    @GetMapping("/stats")
    public Result<OrderStatsVO> getStats(
            @Parameter(description = "开始时间(yyyy-MM-dd HH:mm:ss)") @RequestParam(required = false) @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime startTime,
            @Parameter(description = "结束时间(yyyy-MM-dd HH:mm:ss)") @RequestParam(required = false) @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime endTime) {
        return Result.success(orderService.getStats(startTime, endTime));
    }
}
