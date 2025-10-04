package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.service.SysDatasetItemService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@Tag(name = "08.数据集接口")
@RestController
@RequestMapping("/api/v1/dataset/item")
@RequiredArgsConstructor
public class SysDatasetItemController {

    private final SysDatasetItemService sysDatasetItemService;

    @PostMapping
    @Operation(summary = "新增数据项")
    public Result<Long> addItem(
            @Parameter(description = "所属数据集ID") @RequestParam(value = "datasetId") Long datasetId,
            @Parameter(description = "名称") @RequestParam(value = "name", required = false) String name
    ) {
        SysDatasetItem datasetItem = sysDatasetItemService.createDatasetItem(datasetId, name);
        return Result.success(datasetItem.getId());
    }

    @PutMapping
    @Operation(summary = "修改数据项")
    public Result<Void> updateItem(
            @Parameter(description = "数据项ID") @RequestParam(value = "datasetItemId") Long datasetItemId,
            @Parameter(description = "名称") @RequestParam(value = "name", required = false) String name
    ) {
        sysDatasetItemService.updateDatasetItem(datasetItemId, name);
        return Result.success();
    }

    @DeleteMapping
    @Operation(summary = "删除数据项")
    public Result<Void> removeItem(@Parameter(description = "数据项ID") @RequestParam(value = "datasetItemId") Long datasetItemId) {
        sysDatasetItemService.deleteDatasetItem(datasetItemId);
        return Result.success();
    }
}
