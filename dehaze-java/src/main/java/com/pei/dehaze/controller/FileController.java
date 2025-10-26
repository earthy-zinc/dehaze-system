package com.pei.dehaze.controller;

import cn.hutool.core.date.DateUtil;
import cn.hutool.core.io.FileUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.common.util.FileUploadUtils;
import com.pei.dehaze.model.bo.FileBO;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.service.SysFileService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.InputStreamResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDateTime;

@Tag(name = "07.文件接口")
@RestController
@RequestMapping("/api/v1/files")
@RequiredArgsConstructor
@Slf4j
public class FileController {

    private final SysFileService sysFileService;

    @Value("${file.baseUrl}")
    private String baseUrl;

    @PostMapping
    @Operation(summary = "文件上传")
    public Result<SysFile> uploadFile(
            @Parameter(description = "表单文件对象") @RequestParam(value = "file") MultipartFile file,
            @Parameter(description = "模型id") @RequestParam(required = false) Long modelId
    ) {
        String uploadPath = "upload/" + DateUtil.format(LocalDateTime.now(), "yyyyMMdd");
        FileBO fileBO = FileUploadUtils.createFileBO(file, baseUrl, uploadPath);
        SysFile fileInfo = sysFileService.saveFile(fileBO);
        if (modelId != null) {
            SysFile wpxFile = sysFileService.getWpxFile(fileInfo, modelId);
            return Result.success(wpxFile);
        }
        return Result.success(fileInfo);
    }

    @DeleteMapping
    @Operation(summary = "文件删除")
    @SneakyThrows
    public Result<Void> deleteFile(
            @Parameter(description = "文件路径") @RequestParam Long fileId
    ) {
        boolean result = sysFileService.deleteFile(fileId);
        return Result.judge(result);
    }

    @GetMapping("/check")
    @Operation(summary = "文件校验")
    public Result<Boolean> checkFile(
            @Parameter(description = "文件md5") @RequestParam String md5
    ) {
        boolean result = sysFileService.check(md5);
        return Result.judge(result);
    }

    @GetMapping("/{fileId}")
    @Operation(summary = "获取文件详情")
    public Result<SysFile> getFileDetail(
            @Parameter(description = "文件ID") @PathVariable Long fileId
    ) {
        SysFile file = sysFileService.getById(fileId);
        return Result.success(file);
    }

    @GetMapping("/page")
    @Operation(summary = "分页查询文件")
    public Result<PageResult<SysFile>> listPagedFiles(
            @Parameter(description = "页码") @RequestParam(defaultValue = "1") Integer pageNum,
            @Parameter(description = "每页数量") @RequestParam(defaultValue = "10") Integer pageSize,
            @Parameter(description = "关键字") @RequestParam(required = false) String keywords
    ) {
        Page<SysFile> page = new Page<>(pageNum, pageSize);
        LambdaQueryWrapper<SysFile> queryWrapper = new LambdaQueryWrapper<>();
        if (keywords != null && !keywords.isEmpty()) {
            queryWrapper.like(SysFile::getName, keywords)
                    .or()
                    .like(SysFile::getType, keywords);
        }
        Page<SysFile> result = sysFileService.page(page, queryWrapper);
        return Result.success(PageResult.success(result));
    }


    @GetMapping("/download/**")
    @Operation(summary = "文件下载")
    public ResponseEntity<Resource> download(HttpServletRequest request) {
        String fullPath = request.getRequestURI();
        String objectName = fullPath.substring("/api/v1/files/download/".length());
        String filename = FileUtil.getName(objectName);
        HttpHeaders headers = new HttpHeaders();
        headers.add(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + filename + "\"");
        InputStreamResource resource = new InputStreamResource(sysFileService.download(objectName));
        return ResponseEntity.ok()
                .headers(headers)
                .body(resource);
    }

}