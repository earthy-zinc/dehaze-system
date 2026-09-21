package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.PromotionForm;
import com.pei.dehaze.model.form.PromotionPackageForm;
import com.pei.dehaze.model.query.PromotionPageQuery;
import com.pei.dehaze.model.vo.PromotionVO;
import com.pei.dehaze.service.PromotionService;
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

@Tag(name = "13.促销活动管理")
@RestController
@RequestMapping("/api/v1/packages/promotions")
@RequiredArgsConstructor
public class PromotionController {

    private final PromotionService promotionService;

    @Operation(summary = "后台：促销活动分页列表")
    @GetMapping("/page")
    public PageResult<PromotionVO> getPage(@Valid @ParameterObject PromotionPageQuery query) {
        IPage<PromotionVO> page = promotionService.getPage(query);
        return PageResult.success(page);
    }

    @Operation(summary = "后台：创建促销活动")
    @PostMapping
    @PreAuthorize("@ss.hasPerm('package:promotion:add')")
    public Result<PromotionVO> add(@Valid @RequestBody PromotionForm form) {
        return Result.success(promotionService.save(form));
    }

    @Operation(summary = "后台：修改促销活动")
    @PutMapping("/{id}")
    @PreAuthorize("@ss.hasPerm('package:promotion:edit')")
    public Result<PromotionVO> update(@Parameter(description = "活动ID") @PathVariable Long id,
                                      @Valid @RequestBody PromotionForm form) {
        return Result.success(promotionService.update(id, form));
    }

    @Operation(summary = "后台：促销活动上架/下架")
    @PutMapping("/{id}/status")
    @PreAuthorize("@ss.hasPerm('package:promotion:edit')")
    public Result<PromotionVO> updateStatus(@Parameter(description = "活动ID") @PathVariable Long id,
                                            @Parameter(description = "状态(1:启用;0:禁用)", required = true) @RequestParam Integer status) {
        return Result.success(promotionService.updateStatus(id, status));
    }

    @Operation(summary = "后台：删除促销活动")
    @DeleteMapping("/{id}")
    @PreAuthorize("@ss.hasPerm('package:promotion:delete')")
    public Result<Void> delete(@Parameter(description = "活动ID") @PathVariable Long id) {
        promotionService.delete(id);
        return Result.success();
    }

    @Operation(summary = "后台：关联套餐")
    @PutMapping("/{id}/packages")
    @PreAuthorize("@ss.hasPerm('package:promotion:edit')")
    public Result<Void> bindPackages(@Parameter(description = "活动ID") @PathVariable Long id,
                                     @Valid @RequestBody PromotionPackageForm form) {
        promotionService.bindPackages(id, form);
        return Result.success();
    }
}
