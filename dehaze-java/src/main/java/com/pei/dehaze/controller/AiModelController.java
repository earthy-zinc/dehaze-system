package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AiModelForm;
import com.pei.dehaze.model.form.AiModelUpdateForm;
import com.pei.dehaze.model.form.ModelPriceForm;
import com.pei.dehaze.model.form.ModelPriceUpdateForm;
import com.pei.dehaze.model.query.AiModelPageQuery;
import com.pei.dehaze.model.query.ModelPriceQuery;
import com.pei.dehaze.model.vo.AiModelVO;
import com.pei.dehaze.model.vo.ModelPriceVO;
import com.pei.dehaze.service.AiModelService;
import io.swagger.v3.oas.annotations.Operation;
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

@Tag(name = "AI模型管理")
@RestController
@RequestMapping("/api/v1/ai/models")
@RequiredArgsConstructor
public class AiModelController {

    private final AiModelService aiModelService;

    @Operation(summary = "模型分页列表")
    @GetMapping
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public PageResult<AiModelVO> listModels(@Valid @ParameterObject AiModelPageQuery query) {
        Page<AiModelVO> page = aiModelService.listModels(query);
        return PageResult.success(page);
    }

    @Operation(summary = "启用模型列表")
    @GetMapping("/enabled")
    public Result<List<AiModelVO>> listEnabledModels(
            @RequestParam(required = false) String modelType) {
        return Result.success(aiModelService.listEnabledModels(modelType));
    }

    @Operation(summary = "新增模型")
    @PostMapping
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<AiModelVO> createModel(@RequestBody @Valid AiModelForm form) {
        return Result.success(aiModelService.createModel(form));
    }

    @Operation(summary = "更新模型")
    @PutMapping("/{modelId}")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<AiModelVO> updateModel(@PathVariable String modelId,
                                         @RequestBody @Valid AiModelUpdateForm form) {
        return Result.success(aiModelService.updateModel(modelId, form));
    }

    @Operation(summary = "删除模型")
    @DeleteMapping("/{modelId}")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<Void> deleteModel(@PathVariable String modelId) {
        aiModelService.deleteModel(modelId);
        return Result.success();
    }

    @Operation(summary = "模型用户售价版本分页列表")
    @GetMapping("/{modelId}/prices")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public PageResult<ModelPriceVO> listModelPrices(@PathVariable String modelId,
                                                    @Valid @ParameterObject ModelPriceQuery query) {
        query.setModelId(modelId);
        return PageResult.success(aiModelService.listPrices(query));
    }

    @Operation(summary = "新增模型用户售价版本")
    @PostMapping("/{modelId}/prices")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<ModelPriceVO> createModelPrice(@PathVariable String modelId,
                                                 @RequestBody @Valid ModelPriceForm form) {
        return Result.success(aiModelService.createPrice(form));
    }

    @Operation(summary = "更新模型用户售价版本")
    @PutMapping("/{modelId}/prices/{priceId}")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<ModelPriceVO> updateModelPrice(@PathVariable String modelId,
                                                 @PathVariable Long priceId,
                                                 @RequestBody @Valid ModelPriceUpdateForm form) {
        return Result.success(aiModelService.updatePrice(priceId, form));
    }

    @Operation(summary = "删除模型用户售价版本")
    @DeleteMapping("/{modelId}/prices/{priceId}")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<Void> deleteModelPrice(@PathVariable String modelId,
                                         @PathVariable Long priceId) {
        aiModelService.deletePrice(priceId);
        return Result.success();
    }
}
