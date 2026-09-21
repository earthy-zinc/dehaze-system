package com.pei.dehaze.controller;

import cn.hutool.core.util.StrUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.common.result.ResultCode;
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
import org.springframework.security.access.prepost.PreAuthorize;
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
    @PreAuthorize("@ss.hasPerm('sys:dataset:edit')")
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
        // 上传安全校验链（对齐 python：类型枚举 → 扩展名白名单 → 内容有效性，全部 A0400）
        java.util.Set<String> itemFileTypes = java.util.Set.of("clear", "hazy", "trans", "depth", "segment");
        if (form.getType() == null || !itemFileTypes.contains(form.getType())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "图片类型仅支持 clear/hazy/trans/depth/segment");
        }
        String originalName = form.getFile() != null ? form.getFile().getOriginalFilename() : null;
        if (StrUtil.isBlank(originalName)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "文件名不能为空");
        }
        String ext = originalName.contains(".") ? originalName.substring(originalName.lastIndexOf('.') + 1).toLowerCase() : "";
        java.util.Set<String> imageExtensions = java.util.Set.of("jpg", "jpeg", "png", "gif", "bmp", "webp");
        if (!imageExtensions.contains(ext)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "仅支持图片格式（jpg/png/gif/bmp/webp）");
        }
        try {
            byte[] head = new byte[12];
            int read = form.getFile().getInputStream().read(head);
            boolean valid = validateMagic(ext, head, read);
            if (!valid) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "文件内容不是有效的图片");
            }
        } catch (java.io.IOException e) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "文件内容不是有效的图片");
        }

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
    @PreAuthorize("@ss.hasPerm('sys:dataset:edit')")
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
    @PreAuthorize("@ss.hasPerm('sys:dataset:delete')")
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
    @PreAuthorize("@ss.hasPerm('sys:dataset:delete')")
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

    /** 图片扩展名 → 文件头魔数一致性校验（python validate_image_magic_bytes 同款） */
    private boolean validateMagic(String ext, byte[] head, int read) {
        if (read < 4) {
            return false;
        }
        return switch (ext) {
            case "jpg", "jpeg" -> (head[0] & 0xFF) == 0xFF && (head[1] & 0xFF) == 0xD8 && (head[2] & 0xFF) == 0xFF;
            case "png" -> (head[0] & 0xFF) == 0x89 && head[1] == 'P' && head[2] == 'N' && head[3] == 'G';
            case "gif" -> head[0] == 'G' && head[1] == 'I' && head[2] == 'F';
            case "bmp" -> head[0] == 'B' && head[1] == 'M';
            case "webp" -> head[0] == 'R' && head[1] == 'I' && head[2] == 'F' && head[3] == 'F'
                    && head[8] == 'W' && head[9] == 'E' && head[10] == 'B' && head[11] == 'P';
            default -> true;
        };
    }
}
