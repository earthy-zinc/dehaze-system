package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.common.util.FileDTOFactory;
import com.pei.dehaze.model.dto.ItemFileDTO;
import com.pei.dehaze.model.form.BatchDeleteForm;
import com.pei.dehaze.model.form.ItemFileUploadForm;
import com.pei.dehaze.model.form.ItemFileUpdateForm;
import com.pei.dehaze.model.vo.BatchDeleteResultVO;
import com.pei.dehaze.model.vo.ImageUrlVO;
import com.pei.dehaze.service.SysDatasetService;
import com.pei.dehaze.service.SysItemFileService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;


@Tag(name = "08.图片文件接口")
@RestController
@RequestMapping("/api/v1/item-files")
@RequiredArgsConstructor
public class SysItemFileController {

    private final SysItemFileService sysItemFileService;
    private final SysDatasetService sysDatasetService;
    private final FileDTOFactory fileDTOFactory;

    @GetMapping("/{id}")
    @Operation(
            summary = "获取图片详细信息",
            description = "根据图片ID获取图片的完整信息，包括图片URL、缩略图URL、分辨率（宽×高）、" +
                    "文件大小、文件格式、场景类型、雾霾程度、使用次数等详细数据。" +
                    "同时返回配对图片列表和所属数据项简要信息。适用于图片详情页展示、配对图片切换等场景。"
    )
    public Result<ImageUrlVO> getImageById(
            @Parameter(description = "图片ID", required = true, example = "1")
            @PathVariable
            Long id
    ) {
        ImageUrlVO detail = sysItemFileService.getImageById(id);
        return Result.success(detail);
    }

    @PostMapping
    @Operation(
            summary = "上传数据项图片",
            description = "向指定的数据项添加图片文件，支持上传清晰图或有雾图。" +
                    "系统会自动解析图片宽高、生成缩略图、计算文件MD5。" +
                    "支持添加场景类型、雾霾程度等标注信息。适用于分步骤上传、补充配对图片等场景。"
    )
    public Result<ImageUrlVO> upload(
            @Parameter(description = "图片上传表单数据", content = @Content(mediaType = "multipart/form-data"))
            @Valid @ModelAttribute
            ItemFileUploadForm form
    ) {
        // 获取数据集名称用于存储路径
        String datasetName = sysDatasetService.getDatasetNameByItemId(form.getItemId());
        ItemFileDTO itemBO = fileDTOFactory.createItemFileDTO(
                form.getFile(), datasetName,
                form.getType(), form.getDescription(), form.getSceneType(), form.getHazeLevel()
        );

        ImageUrlVO imageInfo = sysItemFileService.saveItemFile(form.getItemId(), itemBO);
        return Result.success(imageInfo);
    }

    @PutMapping("/{id}")
    @Operation(
            summary = "修改图片信息",
            description = "更新图片的标注信息，支持修改图片类型、场景类型、雾霾程度、描述等字段。" +
                    "不支持修改图片文件本身，如需更换图片请删除后重新上传。" +
                    "系统自动更新修改时间。适用于图片标注、信息完善等场景。"
    )
    public Result<Void> update(
            @Parameter(description = "图片ID", required = true, example = "1")
            @PathVariable
            Long id,
            @Valid @RequestBody
            ItemFileUpdateForm form
    ) {
        boolean result = sysItemFileService.updateItemFileInfo(id, form);
        return Result.judge(result);
    }

    @DeleteMapping("/{id}")
    @Operation(
            summary = "删除图片",
            description = "删除指定的图片文件，同时删除对应的缩略图文件。" +
                    "如果删除的是配对图片中的一张，不会影响其他配对图片。" +
                    "删除操作不可逆，请谨慎使用。"
    )
    public Result<Void> delete(
            @Parameter(description = "图片ID", required = true, example = "1")
            @PathVariable
            Long id
    ) {
        boolean result = sysItemFileService.deleteFile(id);
        return Result.judge(result);
    }

    @DeleteMapping("/batch")
    @Operation(
            summary = "批量删除图片",
            description = "批量删除指定的图片文件，同时删除对应的缩略图文件。" +
                    "支持一次最多删除100张图片。" +
                    "返回删除成功的ID列表和删除失败的ID列表及失败原因。" +
                    "删除操作不可逆，请谨慎使用。"
    )
    public Result<BatchDeleteResultVO> batchDelete(
            @Valid @RequestBody
            BatchDeleteForm form
    ) {
        BatchDeleteResultVO result = sysItemFileService.batchDelete(form.getIds());
        return Result.success(result);
    }
}
