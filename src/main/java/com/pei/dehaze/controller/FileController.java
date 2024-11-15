package com.pei.dehaze.controller;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.dto.FileInfo;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.impl.file.LocalFileService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.Resource;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Tag(name = "07.文件接口")
@RestController
@RequestMapping("/api/v1/files")
@RequiredArgsConstructor
@Slf4j
public class FileController {

    private final FileService fileService;

    @PostMapping
    @Operation(summary = "文件上传")
    public Result<SysFile> uploadFile(
            @Parameter(description ="表单文件对象") @RequestParam(value = "file") MultipartFile file
    ) {
        SysFile fileInfo = fileService.uploadFile(file);
        return Result.success(fileInfo);
    }

    @DeleteMapping
    @Operation(summary = "文件删除")
    @SneakyThrows
    public Result<Void> deleteFile(
            @Parameter(description ="文件路径") @RequestParam String filePath
    ) {
        boolean result = fileService.deleteFile(filePath);
        return Result.judge(result);
    }

    @GetMapping("/check")
    @Operation(summary = "文件校验")
    public Result<Boolean> checkFile(
            @Parameter(description = "文件md5") @RequestParam String md5
    ) {
        boolean result = fileService.uploadCheck(md5);
        return Result.success(result);
    }

    /**
     * @see com.pei.dehaze.filter.FileDownloadFilter
     */
    @GetMapping("/{filePath}")
    @Operation(summary = "文件下载")
    public ResponseEntity<Resource> download(@Parameter(description = "文件路径") @PathVariable String filePath) {
        if (fileService instanceof LocalFileService) {
            // 实际的实现逻辑在FileDownloadFilter拦截器中
            return null;
        } else {
            throw new BusinessException("未开启本地文件存储服务，请检查application.yml文件");
        }
    }

}
