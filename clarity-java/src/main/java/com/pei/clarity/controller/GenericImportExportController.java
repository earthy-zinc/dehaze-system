package com.pei.clarity.controller;

import com.pei.clarity.common.result.Result;
import com.pei.clarity.model.form.ExportPostRequest;
import com.pei.clarity.model.form.ExportRequest;
import com.pei.clarity.model.form.ImportRequest;
import com.pei.clarity.plugin.dupsubmit.annotation.PreventDuplicateSubmit;
import com.pei.clarity.service.importexport.ImportExportService;
import com.pei.clarity.service.importexport.TemplateManager;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * 通用导入导出控制器
 * <p>统一处理所有模块的导入导出请求，通过路径变量 {module} 区分模块。
 * <p>支持的模块：user / role / dept / menu / dict / dataset（仅导出）
 */
@Tag(name = "11.通用导入导出接口")
@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor
public class GenericImportExportController {

    private final ImportExportService importExportService;
    private final TemplateManager templateManager;

    @Operation(summary = "导出数据（GET，简单查询条件）", description = "通过查询参数传递筛选条件，同步导出返回文件流，异步导出返回 taskId")
    @GetMapping("/{module}/_export")
    @PreAuthorize("@ss.hasPerm('sys:' + #module + ':export')")
    public Object export(
            @Parameter(description = "模块标识：user/role/dept/menu/dict/dataset") @PathVariable String module,
            ExportRequest req,
            HttpServletRequest httpRequest,
            HttpServletResponse response) {
        Map<String, Object> queryParams = req.toQueryParams(httpRequest.getParameterMap());
        Object result = importExportService.export(
                module, queryParams, req.getFormat(), req.getAsync(), req.getFieldList(), response);
        return result != null ? Result.success(result) : null;
    }

    @Operation(summary = "导出数据（POST，复杂查询条件）", description = "通过请求体传递复杂查询条件，同步导出返回文件流，异步导出返回 taskId")
    @PostMapping("/{module}/_export")
    @PreAuthorize("@ss.hasPerm('sys:' + #module + ':export')")
    public Object exportPost(
            @Parameter(description = "模块标识：user/role/dept/menu/dict/dataset") @PathVariable String module,
            @RequestBody ExportPostRequest req,
            HttpServletResponse response) {
        Object result = importExportService.export(
                module, req.getQueryParams(), req.getFormat(), req.getAsync(), req.getFields(), response);
        return result != null ? Result.success(result) : null;
    }

    @Operation(summary = "导入数据", description = "上传 Excel/CSV 文件导入数据，同步导入返回结果，异步导入返回 taskId")
    @PostMapping("/{module}/_import")
    @PreAuthorize("@ss.hasPerm('sys:' + #module + ':import')")
    @PreventDuplicateSubmit
    public Result<Object> importData(
            @Parameter(description = "模块标识：user/role/dept/menu/dict") @PathVariable String module,
            ImportRequest req,
            @RequestParam("file") MultipartFile file,
            HttpServletRequest httpRequest) {
        Map<String, Object> extraParams = req.toExtraParams(httpRequest.getParameterMap());
        Object result =
                importExportService.importData(module, file, req.getModeOrDefault(), req.getAsync(), extraParams);
        return Result.success(result);
    }

    @Operation(summary = "下载导入模板", description = "动态生成包含表头和示例数据的导入模板，支持 Excel 和 CSV 格式")
    @GetMapping("/{module}/template")
    @PreAuthorize("@ss.hasPerm('sys:' + #module + ':import')")
    public void downloadTemplate(
            @Parameter(description = "模块标识：user/role/dept/menu/dict") @PathVariable String module,
            @Parameter(description = "文件格式：excel(默认) / csv") @RequestParam(defaultValue = "excel") String format,
            HttpServletResponse response) {
        templateManager.downloadTemplate(module, format, response);
    }
}
