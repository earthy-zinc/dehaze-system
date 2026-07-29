package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.CouponBatchDistributeForm;
import com.pei.dehaze.model.form.CouponForm;
import com.pei.dehaze.model.form.PackageForm;
import com.pei.dehaze.model.query.CouponPageQuery;
import com.pei.dehaze.model.query.PackagePageQuery;
import com.pei.dehaze.model.vo.CouponBatchResult;
import com.pei.dehaze.model.vo.CouponCreateResult;
import com.pei.dehaze.model.vo.CouponReceiveResult;
import com.pei.dehaze.model.vo.CouponVO;
import com.pei.dehaze.model.vo.PackageDetailVO;
import com.pei.dehaze.model.vo.PackagePageVO;
import com.pei.dehaze.model.vo.PriceResult;
import com.pei.dehaze.model.vo.SalesStatsVO;
import com.pei.dehaze.model.vo.UserCouponVO;
import com.pei.dehaze.service.CouponService;
import com.pei.dehaze.service.PackageService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Tag(name = "12.套餐管理")
@RestController
@RequestMapping("/api/v1/packages")
@RequiredArgsConstructor
public class PackageController {

    private final PackageService packageService;
    private final CouponService couponService;

    @Operation(summary = "用户端：在售套餐列表")
    @GetMapping
    public Result<List<PackageDetailVO>> listOnSale() {
        return Result.success(packageService.listOnSale());
    }

    @Operation(summary = "用户端：套餐详情")
    @GetMapping("/{id}")
    public Result<PackageDetailVO> getDetail(@Parameter(description = "套餐ID") @PathVariable Long id) {
        return Result.success(packageService.getDetail(id));
    }

    @Operation(summary = "价格计算（下单前预览）")
    @GetMapping("/calculate-price")
    public Result<PriceResult> calculatePrice(
            @Parameter(description = "套餐ID", required = true) @RequestParam Long packageId,
            @Parameter(description = "用户优惠券实例ID") @RequestParam(required = false) Long userCouponId) {
        return Result.success(packageService.calculatePrice(packageId, userCouponId));
    }

    @Operation(summary = "后台：套餐分页列表")
    @GetMapping("/page")
    public PageResult<PackagePageVO> getPage(@ParameterObject PackagePageQuery query) {
        Page<PackagePageVO> page = packageService.getPage(query);
        return PageResult.success(page);
    }

    @Operation(summary = "后台：获取套餐表单数据")
    @GetMapping("/{id}/form")
    public Result<PackageForm> getForm(@Parameter(description = "套餐ID") @PathVariable Long id) {
        return Result.success(packageService.getForm(id));
    }

    @Operation(summary = "后台：新增套餐")
    @PostMapping
    @PreAuthorize("@ss.hasPerm('package:add')")
    public Result<Void> add(@Valid @RequestBody PackageForm form) {
        packageService.save(form);
        return Result.success();
    }

    @Operation(summary = "后台：修改套餐")
    @PutMapping("/{id}")
    @PreAuthorize("@ss.hasPerm('package:edit')")
    public Result<Void> update(@Parameter(description = "套餐ID") @PathVariable Long id,
                               @Valid @RequestBody PackageForm form) {
        packageService.update(id, form);
        return Result.success();
    }

    @Operation(summary = "后台：上架/下架")
    @PutMapping("/{id}/status")
    @PreAuthorize("@ss.hasPerm('package:edit')")
    public Result<Void> updateStatus(@Parameter(description = "套餐ID") @PathVariable Long id,
                                     @Parameter(description = "状态(1:上架;0:下架)", required = true) @RequestParam Integer status) {
        packageService.updateStatus(id, status);
        return Result.success();
    }

    @Operation(summary = "后台：删除套餐")
    @DeleteMapping("/{ids}")
    @PreAuthorize("@ss.hasPerm('package:delete')")
    public Result<Void> deleteByIds(@Parameter(description = "套餐ID（逗号分隔）") @PathVariable String ids) {
        packageService.deleteByIds(ids);
        return Result.success();
    }

    @Operation(summary = "后台：销售统计")
    @GetMapping("/sales/stats")
    @PreAuthorize("@ss.hasPerm('package:sales')")
    public Result<SalesStatsVO> getSalesStats() {
        return Result.success(packageService.getSalesStats());
    }

    // ============ 优惠券管理 ============

    @Operation(summary = "用户端：我的优惠券列表")
    @GetMapping("/coupons/my")
    public Result<List<UserCouponVO>> listMyCoupons(
            @Parameter(description = "状态(1:未使用;2:已使用;3:已过期;4:已锁定)") @RequestParam(required = false) Integer status) {
        return Result.success(couponService.listMy(status));
    }

    @Operation(summary = "用户端：领取优惠券")
    @PostMapping("/coupons/{couponId}/receive")
    public Result<CouponReceiveResult> receiveCoupon(@Parameter(description = "优惠券ID") @PathVariable Long couponId) {
        return Result.success(couponService.receive(couponId));
    }

    @Operation(summary = "后台：优惠券分页列表")
    @GetMapping("/coupons/page")
    public PageResult<CouponVO> getCouponPage(@ParameterObject CouponPageQuery query) {
        Page<CouponVO> page = couponService.getPage(query);
        return PageResult.success(page);
    }

    @Operation(summary = "后台：创建优惠券")
    @PostMapping("/coupons")
    @PreAuthorize("@ss.hasPerm('package:coupon:add')")
    public Result<CouponCreateResult> addCoupon(@Valid @RequestBody CouponForm form) {
        return Result.success(couponService.create(form));
    }

    @Operation(summary = "后台：批量发放优惠券")
    @PostMapping("/coupons/batch")
    @PreAuthorize("@ss.hasPerm('package:coupon:distribute')")
    public Result<CouponBatchResult> batchDistribute(@Valid @RequestBody CouponBatchDistributeForm form) {
        return Result.success(couponService.batchDistribute(form));
    }

    @Operation(summary = "后台：修改优惠券")
    @PutMapping("/coupons/{id}")
    @PreAuthorize("@ss.hasPerm('package:coupon:edit')")
    public Result<Void> updateCoupon(@Parameter(description = "优惠券ID") @PathVariable Long id,
                                     @Valid @RequestBody CouponForm form) {
        couponService.update(id, form);
        return Result.success();
    }

    @Operation(summary = "后台：删除优惠券")
    @DeleteMapping("/coupons/{ids}")
    @PreAuthorize("@ss.hasPerm('package:coupon:delete')")
    public Result<Void> deleteCoupons(@Parameter(description = "优惠券ID（逗号分隔）") @PathVariable String ids) {
        couponService.deleteByIds(ids);
        return Result.success();
    }
}
