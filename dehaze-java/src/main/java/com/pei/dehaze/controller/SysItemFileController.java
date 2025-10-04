package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.common.util.FileUploadUtils;
import com.pei.dehaze.model.bo.DatasetItemBO;
import com.pei.dehaze.model.dto.ImageFileInfo;
import com.pei.dehaze.service.SysDatasetService;
import com.pei.dehaze.service.SysItemFileService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;


@Tag(name = "08.数据集接口")
@RestController
@RequestMapping("/api/v1/dataset/image")
@RequiredArgsConstructor
public class SysItemFileController {

    private final SysItemFileService sysItemFileService;

    private final SysDatasetService sysDatasetService;

    @Value("${file.baseUrl}")
    private String baseUrl;

    /**
     * 向当前数据项中上传图片以及对应信息
     *
     * @param file 图片文件
     * @return 文件信息
     */
    @PostMapping
    @Operation(summary = "上传数据项图片")
    public Result<ImageFileInfo> addImageById(
            @Parameter(description = "表单文件对象") @RequestParam(value = "file") MultipartFile file,
            @Parameter(description = "所属数据集ID") @RequestParam(value = "datasetId") Long datasetId,
            @Parameter(description = "所属数据项ID") @RequestParam(value = "datasetItemId") Long datasetItemId,
            @Parameter(description = "图片类型") @RequestParam(value = "type") String type,
            @Parameter(description = "图片描述") @RequestParam(value = "description", required = false) String description
    ) {
        String datasetName = sysDatasetService.getRootDataset(datasetId).getName();
        DatasetItemBO itemBO = FileUploadUtils.createDatasetItemBO(file, baseUrl, datasetName, type, description);
        ImageFileInfo imageInfo = sysItemFileService.saveItemFile(datasetItemId, itemBO);
        return Result.success(imageInfo);
    }

    /**
     * 修改当前数据项中某个图片的对应信息
     *
     * @return 文件信息
     */
    @PutMapping
    @Operation(summary = "修改数据项图片信息")
    public Result<ImageFileInfo> updateImageById(
            @Parameter(description = "数据项文件ID") @RequestParam(value = "itemFileId") Long itemFileId,
            @Parameter(description = "图片类型") @RequestParam(value = "type") String type,
            @Parameter(description = "图片描述") @RequestParam(value = "description", required = false) String description
    ) {
        return Result.failed();
    }

    @DeleteMapping
    @Operation(summary = "删除数据项图片")
    public Result<Void> removeImageById(@Parameter(description = "数据项文件ID") @RequestParam(value = "itemFileId") Long itemFileId) {
        boolean result = sysItemFileService.deleteItemFile(itemFileId);
        return Result.judge(result);
    }
}
