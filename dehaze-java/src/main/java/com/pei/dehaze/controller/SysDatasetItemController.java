package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.form.*;
import com.pei.dehaze.model.query.DatasetItemQuery;
import com.pei.dehaze.model.vo.*;
import com.pei.dehaze.service.DownloadService;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysItemFileService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotEmpty;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;

@Tag(name = "08.数据集接口")
@RestController
@RequestMapping("/api/v1/dataset/item")
@RequiredArgsConstructor
public class SysDatasetItemController {

    private final SysDatasetItemService sysDatasetItemService;

    private final SysItemFileService sysItemFileService;

    private final DownloadService downloadService;

    @GetMapping("/{id}")
    @Operation(summary = "获取数据项详情")
    public Result<DatasetItemVO> getItemDetailById(@PathVariable Long id) {
        DatasetItemVO detail = sysDatasetItemService.getDatasetItem(id);
        return Result.success(detail);
    }

    @Operation(summary = "获取数据集图片详细信息")
    @GetMapping
    public PageResult<DatasetItemVO> getDatasetImageDetail(@ParameterObject DatasetItemQuery query) {
        Page<DatasetItemVO> result = sysDatasetItemService.pageSearchDatasetItems(query);
        return PageResult.success(result);
    }

    @PostMapping
    @Operation(
            summary = "新增数据项",
            description = "仅创建数据项，未在数据项中添加图片和详细信息，如需对数据项进一步完善，则需要调用 上传数据项图片 API进一步完善"
    )
    public Result<Long> addItem(
            @Parameter(description = "所属数据集ID") @RequestParam(value = "datasetId") Long datasetId,
            @Parameter(description = "名称") @RequestParam(value = "name", required = false) String name
    ) {
        SysDatasetItem datasetItem = sysDatasetItemService.createDatasetItem(datasetId, name);
        return Result.success(datasetItem.getId());
    }

    @PostMapping("/upload")
    @Operation(summary = "创建数据项并上传图片")
    public Result<DatasetItemVO> uploadImagePair(@Valid @ModelAttribute DatasetItemUploadForm form) {
        DatasetItemVO result = sysItemFileService.createDatasetItemAndUpload(form);
        return Result.success(result);
    }

    @PostMapping("/batch/upload")
    @Operation(
            summary = "批量创建数据项并上传图片",
            description = "该方法会根据图片名称，自动的对同一有雾/无雾图片进行配对组成数据项，进行批量的创建"
    )
    public Result<BatchUploadResultVO> batchUploadImagePairs(@Valid @ModelAttribute BatchDatasetItemUploadForm form) {
        BatchUploadResultVO result = sysItemFileService.batchCreateDatasetItemAndUpload(form);
        return Result.success(result);
    }

    @PutMapping
    @Operation(summary = "修改数据项")
    public Result<Void> updateItem(
            @Parameter(description = "数据项ID") @RequestParam(value = "datasetItemId") Long datasetItemId,
            @Parameter(description = "名称") @RequestParam(value = "name", required = false) String name,
            @Parameter(description = "场景类型") @RequestParam(value = "sceneType", required = false) String sceneType
    ) {
        sysDatasetItemService.updateDatasetItem(datasetItemId, name);
        return Result.success();
    }

    @DeleteMapping
    @Operation(summary = "删除数据项")
    public Result<Void> removeItem(
            @Parameter(description = "数据项ID")
            @RequestParam(value = "datasetItemId")
            Long datasetItemId
    ) {
        sysDatasetItemService.deleteDatasetItem(datasetItemId);
        return Result.success();
    }

    @DeleteMapping("/batch")
    @Operation(summary = "批量删除数据项")
    public Result<BatchOperationResultVO> batchDeleteItems(
            @Valid
            @RequestBody
            @Parameter(description = "数据项ID列表")
            @NotEmpty(message = "请选择要删除的数据项")
            List<Long> ids
    ) {
        BatchOperationResultVO result = sysDatasetItemService.batchDeleteDatasetItems(ids);
        return Result.success(result);
    }

    @GetMapping("/{id}/download/task")
    @Operation(summary = "下载数据项")
    public Result<DownloadTaskVO> createPairedDownloadTask(
        @PathVariable Long id,
        @Parameter(description = "需要下载的 图片 id，不选则全部下载")
        @RequestParam(value = "itemFileId", required = false)
        ArrayList<String> itemFileIds
    ) {
        // 待实现
        return Result.success(null);
    }

    @PostMapping("/batch/download")
    @Operation(summary = "批量下载数据项")
    public Result<DownloadTaskVO> batchDownloadItems(
            @Valid
            @RequestBody
            @Schema(description = "数据项ID列表", requiredMode = Schema.RequiredMode.REQUIRED)
            @NotEmpty(message = "请选择要下载的数据项")
            List<Long> itemFileIds,
            @RequestBody
            @Schema(description = "是否按数据项分目录组织")
            Boolean organizeByItem
    ) {
        String taskId = downloadService.createBatchImageItemDownloadTask(itemFileIds, organizeByItem);
        DownloadTaskVO task = new DownloadTaskVO();
        task.setTaskId(taskId);
        task.setStatus("processing");
        task.setMessage("正在创建下载任务...");

        return Result.success(task);
    }
}
