package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.common.util.XssUtils;
import com.pei.dehaze.model.form.*;
import com.pei.dehaze.model.query.DatasetItemQuery;
import com.pei.dehaze.model.vo.*;
import com.pei.dehaze.service.DatasetOperationService;
import com.pei.dehaze.service.SysDatasetItemService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.*;

@Tag(name = "08.数据项接口")
@RestController
@RequestMapping("/api/v1/dataset-items")
@RequiredArgsConstructor
public class SysDatasetItemController {

    private final SysDatasetItemService sysDatasetItemService;

    private final DatasetOperationService datasetOperationService;

    @GetMapping("/{id}")
    @Operation(
            summary = "获取数据项详情",
            description = "根据数据项ID获取完整的数据项信息，包括数据项基本信息、场景类型、" +
                    "清晰图信息（Ground Truth）和所有有雾图列表。返回的配对图片信息包含图片URL、" +
                    "分辨率、雾霾程度等详细数据。适用于数据项详情页展示、图片配对查看等场景。"
    )
    public Result<DatasetItemVO> getDatasetItemById(
            @Parameter(description = "数据项ID", required = true, example = "1")
            @PathVariable
            Long id
    ) {
        DatasetItemVO detail = sysDatasetItemService.getDatasetItem(id);
        return Result.success(detail);
    }

    @Operation(
            summary = "分页查询数据项列表",
            description = "根据查询条件分页获取数据项列表，支持多维度筛选：关键字搜索、场景类型、雾霾程度、" +
                    "分辨率范围、文件大小范围等。支持按相关度、创建时间、使用次数排序。" +
                    "适用于数据集详情页的图片列表展示、图片搜索等场景。"
    )
    @GetMapping
    public PageResult<DatasetItemVO> listDatasetItems(@ParameterObject DatasetItemQuery query) {
        Page<DatasetItemVO> result = sysDatasetItemService.pageSearchDatasetItems(query);
        return PageResult.success(result);
    }

    @PostMapping
    @Operation(
            summary = "创建空数据项",
            description = "创建一个空的数据项，仅包含基本信息（名称、所属数据集），不包含图片。" +
                    "创建后需要调用《上传数据项图片》接口添加图片文件。" +
                    "适用于分步骤上传的场景，先创建数据项再逐步添加图片。"
    )
    public Result<DatasetItemVO> addItem(@Valid @RequestBody DatasetItemCreateForm form) {
        // XSS防护：过滤用户输入的名称
        String cleanName = XssUtils.clean(form.getName());
        DatasetItemVO datasetItem = sysDatasetItemService.createAndReturnDatasetItem(
                form.getDatasetId(), cleanName);
        return Result.success(datasetItem);
    }

    @PostMapping("/upload")
    @Operation(
            summary = "创建数据项并上传配对图片",
            description = "一步完成数据项创建和配对图片上传。支持上传一张清晰图（Ground Truth）和多张有雾图。" +
                    "系统会自动校验配对图片的分辨率一致性，自动解析图片宽高，生成缩略图。" +
                    "适用于单个数据项的快速上传。"
    )
    public Result<DatasetItemVO> uploadImagePair(@Valid @ModelAttribute DatasetItemUploadForm form) {
        DatasetItemVO result = datasetOperationService.createDatasetItemWithImages(form);
        return Result.success(result);
    }

    @PostMapping("/batch")
    @Operation(
            summary = "批量创建数据项并上传图片",
            description = "批量上传多个数据项的配对图片，系统根据文件名命名规则自动识别配对关系。" +
                    "命名规则：xxx_clear.jpg/xxx_gt.jpg为清晰图，xxx_hazy_light.jpg为轻度有雾图。" +
                    "同一前缀的图片自动归为一个配对组。支持批量校验分辨率一致性，返回详细的批量处理结果。"
    )
    public Result<BatchUploadResultVO> batchUploadImagePairs(
            @Valid @ModelAttribute BatchDatasetItemUploadForm form
    ) {
        BatchUploadResultVO result = datasetOperationService.batchCreateDatasetItemsWithImages(form);
        return Result.success(result);
    }

    @PutMapping("/{id}")
    @Operation(
            summary = "修改数据项信息",
            description = "更新数据项的基本信息，支持修改数据项名称和场景类型。" +
                    "场景类型会应用到该数据项下的所有图片。系统自动更新修改时间。"
    )
    public Result<DatasetItemVO> updateItem(
            @Parameter(description = "数据项ID", required = true, example = "1") @PathVariable Long id,
            @Valid @RequestBody DatasetItemUpdateForm form
    ) {
        // XSS防护：过滤用户输入的名称和场景类型
        String cleanName = XssUtils.clean(form.getName());
        String cleanSceneType = XssUtils.clean(form.getSceneType());

        DatasetItemVO result = sysDatasetItemService.updateAndReturnDatasetItem(
                id, cleanName, cleanSceneType);
        return Result.success(result);
    }

    @DeleteMapping("/{id}")
    @Operation(
            summary = "删除数据项",
            description = "删除指定的数据项，级联删除该数据项下的所有图片文件（清晰图和有雾图）、" +
                    "缩略图文件。删除操作不可逆，请谨慎使用。"
    )
    public Result<Void> removeItem(
            @Parameter(description = "数据项ID", required = true, example = "1")
            @PathVariable
            Long id
    ) {
        // 使用统一的级联删除入口
        datasetOperationService.deleteDatasetItemCascade(id);
        return Result.success();
    }

    @DeleteMapping("/batch")
    @Operation(
            summary = "批量删除数据项",
            description = "批量删除多个数据项，级联删除所有关联的图片文件和缩略图。" +
                    "返回批量操作结果，包括成功数量、失败数量和失败详情。删除操作不可逆，请谨慎使用。"
    )
    public Result<BatchOperationResultVO> batchDeleteDatasetItems(
            @Valid @RequestBody BatchDeleteForm form
    ) {
        // 使用统一的级联删除入口
        BatchOperationResultVO result = datasetOperationService.batchDeleteDatasetItemsCascadeWithResult(form.getIds());
        return Result.success(result);
    }
}
