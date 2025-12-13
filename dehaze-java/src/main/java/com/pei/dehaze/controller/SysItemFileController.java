package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.common.util.FileUploadUtils;
import com.pei.dehaze.model.bo.ItemFileBO;
import com.pei.dehaze.model.dto.ImageFileInfo;
import com.pei.dehaze.model.form.*;
import com.pei.dehaze.model.vo.ImageDetailVO;
import com.pei.dehaze.service.SysDatasetService;
import com.pei.dehaze.service.SysItemFileService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;


@Tag(name = "08.数据集接口")
@RestController
@RequestMapping("/api/v1/dataset/image")
@RequiredArgsConstructor
public class SysItemFileController {

    private final SysItemFileService sysItemFileService;

    private final SysDatasetService sysDatasetService;


    @Value("${file.baseUrl}")
    private String baseUrl;

    @GetMapping("/{id}")
    @Operation(summary = "获取图片详情")
    public Result<ImageDetailVO> getImageDetail(@PathVariable Long id) {
        ImageDetailVO detail = sysItemFileService.getImageDetail(id);
        return Result.success(detail);
    }

    @PostMapping
    @Operation(summary = "上传数据项图片")
    public Result<ImageFileInfo> addImageById(@Valid @ModelAttribute ItemFileUploadForm form) {
        String datasetName = sysDatasetService.getRootDataset(form.getDatasetId()).getName();
        ItemFileBO itemBO = FileUploadUtils.createItemFileBO(
                form.getFile(), baseUrl, datasetName,
                form.getType(), form.getDescription(), form.getSceneType(), form.getHazeLevel()
        );

        ImageFileInfo imageInfo = sysItemFileService.saveItemFile(form.getDatasetItemId(), itemBO);
        return Result.success(imageInfo);
    }

    @PutMapping
    @Operation(summary = "修改数据项图片信息")
    public Result<Void> updateImageById(@Valid @RequestBody ImageItemForm form) {
        boolean result = sysItemFileService.updateImageItemInfo(form);
        return Result.judge(result);
    }

    @DeleteMapping
    @Operation(summary = "删除数据项图片")
    public Result<Void> removeImageById(@Parameter(description = "数据项文件ID") @RequestParam(value = "itemFileId") Long itemFileId) {
        boolean result = sysItemFileService.deleteItemFile(itemFileId);
        return Result.judge(result);
    }
}
