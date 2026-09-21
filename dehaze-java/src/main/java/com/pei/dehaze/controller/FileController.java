package com.pei.dehaze.controller;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.FileDTOFactory;
import com.pei.dehaze.common.util.FilePathBuilder;
import com.pei.dehaze.model.dto.FileDTO;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.SysFileService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.InputStreamResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.MediaTypeFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;

@Tag(name = "07.文件接口")
@RestController
@RequestMapping("/api/v1/files")
@RequiredArgsConstructor
@Slf4j
public class FileController {

    private final SysFileService sysFileService;
    private final FilePathBuilder filePathBuilder;
    private final FileDTOFactory fileDTOFactory;

    /**
     * 归属校验：管理员全量可见，普通用户仅可访问自己上传的文件（越权 B0407）
     */
    private void ensureFileAccess(SysFile file) {
        if (SecurityUtils.isAdmin()) {
            return;
        }
        if (!SecurityUtils.getUserId().equals(file.getCreateBy())) {
            throw new BusinessException(ResultCode.FILE_ACCESS_DENIED, "无权访问该文件");
        }
    }

    /**
     * sys_file.name / object_name 列宽为 varchar(100)，超长文件名/扩展名会导致入库失败，
     * 三端统一校验口径：文件名 ≤100 字符、扩展名 ≤20 字符
     */
    private void validateFilename(String filename) {
        if (StrUtil.isBlank(filename)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "文件名不能为空");
        }
        if (filename.length() > 100) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "文件名过长");
        }
        String extension = FileUtil.extName(filename);
        if (extension != null && extension.length() > 20) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "文件扩展名过长");
        }
    }

    @PostMapping
    @Operation(summary = "文件上传")
    public Result<SysFile> uploadFile(
            @Parameter(description = "表单文件对象") @RequestParam(value = "file") MultipartFile file,
            @Parameter(description = "模型id") @RequestParam(required = false) Long modelId
    ) {
        validateFilename(file.getOriginalFilename());
        String uploadPath = filePathBuilder.buildUploadPath();
        FileDTO fileDTO = fileDTOFactory.createFileDTO(file, uploadPath);
        SysFile fileInfo = sysFileService.saveFile(fileDTO);
        if (modelId != null) {
            SysFile wpxFile = sysFileService.getWpxFile(fileInfo, modelId);
            sysFileService.fillUrl(wpxFile);
            return Result.success(wpxFile);
        }
        sysFileService.fillUrl(fileInfo);
        return Result.success(fileInfo);
    }

    @DeleteMapping
    @Operation(summary = "文件删除")
    public Result<Void> deleteFile(
            @Parameter(description = "文件路径") @RequestParam Long fileId
    ) {
        SysFile file = sysFileService.getById(fileId);
        if (file == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        ensureFileAccess(file);
        boolean result = sysFileService.deleteFile(fileId);
        return Result.judge(result);
    }

    @GetMapping("/check")
    @Operation(summary = "文件校验")
    public Result<SysFile> checkFile(
            @Parameter(description = "文件md5") @RequestParam String md5
    ) {
        // 校验 MD5 格式：32 位十六进制（T-FM-034/035：无效 MD5 返回 B0404"MD5 格式无效"）
        if (StrUtil.isBlank(md5) || !md5.matches("^[0-9a-fA-F]{32}$")) {
            throw new BusinessException(ResultCode.FILE_MD5_INVALID);
        }
        SysFile fileInfo = sysFileService.check(md5);
        sysFileService.fillUrl(fileInfo);
        return Result.success(fileInfo);
    }

    @GetMapping("/page")
    @Operation(summary = "分页查询文件")
    public PageResult<SysFile> listPagedFiles(
            @Parameter(description = "页码") @RequestParam(defaultValue = "1") Integer pageNum,
            @Parameter(description = "每页数量") @RequestParam(defaultValue = "10") Integer pageSize,
            @Parameter(description = "关键字") @RequestParam(required = false) String keywords
    ) {
        Page<SysFile> page = new Page<>(pageNum, pageSize);
        LambdaQueryWrapper<SysFile> queryWrapper = new LambdaQueryWrapper<>();
        // 普通用户仅可见自己上传的文件，管理员全量
        if (!SecurityUtils.isAdmin()) {
            queryWrapper.eq(SysFile::getCreateBy, SecurityUtils.getUserId());
        }
        if (keywords != null && !keywords.isEmpty()) {
            // 用 and 嵌套包裹 OR，防止 OR 条件逃逸逻辑删除（deleted=0）过滤
            queryWrapper.and(w -> w.like(SysFile::getName, keywords)
                    .or()
                    .like(SysFile::getType, keywords));
        }
        queryWrapper.orderByDesc(SysFile::getCreateTime);
        Page<SysFile> result = sysFileService.page(page, queryWrapper);
        result.getRecords().forEach(sysFileService::fillUrl);
        return PageResult.success(result);
    }

    @GetMapping("/{fileId}")
    @Operation(summary = "获取文件详情")
    public Result<SysFile> getFileDetail(
            @Parameter(description = "文件ID") @PathVariable Long fileId
    ) {
        SysFile file = sysFileService.getById(fileId);
        if (file == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        ensureFileAccess(file);
        sysFileService.fillUrl(file);
        return Result.success(file);
    }

    @GetMapping("/download/**")
    @Operation(summary = "文件下载")
    public ResponseEntity<Resource> download(HttpServletRequest request) {
        String fullPath = request.getRequestURI();
        String objectName = fullPath.substring("/api/v1/files/download/".length());

        // 防止路径遍历攻击（对齐 Python 端校验）
        if (objectName.contains("..") || objectName.startsWith("/") || objectName.contains("\\")) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "无效的文件路径");
        }

        SysFile file = sysFileService.getOne(
                new LambdaQueryWrapper<SysFile>().eq(SysFile::getObjectName, objectName));
        if (file == null) {
            // 不存在/已软删文件返回业务信封（python B0401 口径），不做裸 404
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        ensureFileAccess(file);

        // 统一流式转发：按 storage 选后端，调用 downLoadFile 取流。是否直连存储由 baseUrl 配置决定，不在代码分支
        InputStream stream = sysFileService.download(objectName);
        String filename = FileUtil.getName(objectName);
        HttpHeaders headers = new HttpHeaders();
        headers.add(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + filename + "\"");
        // 根据文件名后缀推断真实 MIME 类型（如 image/jpeg、image/png），避免浏览器/客户端按 application/json 解析图片
        MediaType mediaType = MediaTypeFactory.getMediaType(filename)
                .orElse(MediaType.APPLICATION_OCTET_STREAM);
        headers.setContentType(mediaType);
        return ResponseEntity.ok()
                .headers(headers)
                .body(new InputStreamResource(stream));
    }
}
