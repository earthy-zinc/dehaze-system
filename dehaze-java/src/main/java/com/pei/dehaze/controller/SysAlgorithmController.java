package com.pei.dehaze.controller;

import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.converter.AlgorithmConverter;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysAlgorithmVersion;
import com.pei.dehaze.model.form.AlgorithmAuditForm;
import com.pei.dehaze.model.form.AlgorithmForm;
import com.pei.dehaze.model.form.AlgorithmVersionForm;
import com.pei.dehaze.model.form.ExportRequest;
import com.pei.dehaze.model.form.ImportRequest;
import com.pei.dehaze.model.query.AlgorithmQuery;
import com.pei.dehaze.model.vo.AlgorithmMonitorVO;
import com.pei.dehaze.model.vo.AlgorithmVO;
import com.pei.dehaze.model.vo.AlgorithmVersionVO;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.SysAlgorithmVersionService;
import com.pei.dehaze.service.importexport.ImportExportService;
import com.pei.dehaze.service.importexport.TemplateManager;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;

/**
 * @author earthy-zinc
 * @since 2024-06-08 19:05:51
 */
@Tag(name = "09.算法接口")
@RestController
@RequestMapping("/api/v1/algorithms")
@RequiredArgsConstructor
public class SysAlgorithmController {
    private final SysAlgorithmService algorithmService;
    private final SysAlgorithmVersionService versionService;
    private final AlgorithmConverter algorithmConverter;
    private final ImportExportService importExportService;
    private final TemplateManager templateManager;

    @Operation(summary = "获取算法树形表格")
    @GetMapping
    public Result<List<AlgorithmVO>> getList(@ParameterObject AlgorithmQuery queryParams) {
        List<AlgorithmVO> algorithms = algorithmService.getList(queryParams);
        return Result.success(algorithms);
    }

    @Operation(summary = "获取模型下拉选项列表")
    @GetMapping("/options")
    public Result<List<Option<Long>>> getOption() {
        List<Option<Long>> options = algorithmService.getOption();
        return Result.success(options);
    }

    @Operation(summary = "获取所有算法扁平列表")
    @GetMapping("/list")
    public Result<List<AlgorithmVO>> listAll() {
        List<AlgorithmVO> list = algorithmService.listAll();
        return Result.success(list);
    }

    @Operation(summary = "根据ID获取算法信息")
    @GetMapping("/{id}")
    public Result<AlgorithmVO> getById(@PathVariable Long id) {
        SysAlgorithm algorithm = algorithmService.getAlgorithmById(id);
        return Result.success(algorithmConverter.entity2Vo(algorithm));
    }

    @Operation(summary = "新增算法")
    @PostMapping
    @PreAuthorize("@ss.hasPerm('sys:algorithm:add')")
    public Result<Long> add(@RequestBody @Valid AlgorithmForm algorithm) {
        Long id = algorithmService.addAlgorithm(algorithm);
        return Result.success(id);
    }

    @Operation(summary = "修改算法")
    @PutMapping("/{id}")
    @PreAuthorize("@ss.hasPerm('sys:algorithm:edit')")
    public Result<Void> update(@PathVariable Long id, @RequestBody @Valid AlgorithmForm algorithm) {
        algorithm.setId(id);
        boolean result = algorithmService.updateAlgorithm(algorithm);
        return Result.judge(result);
    }

    @Operation(summary = "删除算法")
    @DeleteMapping
    @PreAuthorize("@ss.hasPerm('sys:algorithm:delete')")
    public Result<Void> deleteByIds(@RequestParam List<Long> ids) {
        boolean result = algorithmService.deleteAlgorithms(ids);
        return Result.judge(result);
    }

    // ==================== 状态管理 ====================

    @Operation(summary = "修改算法状态")
    @PutMapping("/{id}/status")
    @PreAuthorize("@ss.hasPerm('sys:algorithm:edit')")
    public Result<Void> updateStatus(
            @Parameter(description = "算法ID") @PathVariable Long id,
            @RequestBody Map<String, Integer> body) {
        boolean result = algorithmService.updateStatus(id, body.get("status"));
        return Result.judge(result);
    }

    @Operation(summary = "审核算法（通过/驳回）")
    @PutMapping("/{id}/audit")
    @PreAuthorize("@ss.hasPerm('sys:algorithm:audit')")
    public Result<Void> auditAlgorithm(
            @Parameter(description = "算法ID") @PathVariable Long id,
            @Valid @RequestBody AlgorithmAuditForm form) {
        boolean result = algorithmService.auditAlgorithm(id, form);
        return Result.judge(result);
    }

    // ==================== 版本管理 ====================

    @Operation(summary = "获取算法版本历史")
    @GetMapping("/{id}/versions")
    public Result<List<AlgorithmVersionVO>> getVersionHistory(
            @Parameter(description = "算法ID") @PathVariable Long id) {
        List<AlgorithmVersionVO> versions = versionService.getVersionHistory(id);
        return Result.success(versions);
    }

    @Operation(summary = "新增算法版本")
    @PostMapping("/{id}/version")
    @PreAuthorize("@ss.hasPerm('sys:algorithm:version')")
    public Result<SysAlgorithmVersion> addVersion(
            @Parameter(description = "算法ID") @PathVariable Long id,
            @Valid @RequestBody AlgorithmVersionForm form) {
        SysAlgorithmVersion version = versionService.addVersion(id, form);
        return Result.success(version);
    }

    @Operation(summary = "版本回滚")
    @PostMapping("/{id}/rollback")
    @PreAuthorize("@ss.hasPerm('sys:algorithm:version')")
    public Result<Void> rollbackVersion(
            @Parameter(description = "算法ID") @PathVariable Long id,
            @Parameter(description = "目标版本ID") @RequestParam Long versionId) {
        versionService.rollbackToVersion(id, versionId);
        return Result.success();
    }

    // ==================== 性能监控 ====================

    @Operation(summary = "获取算法监控数据")
    @GetMapping("/{id}/monitor")
    public Result<AlgorithmMonitorVO> getMonitorData(
            @Parameter(description = "算法ID") @PathVariable Long id) {
        AlgorithmMonitorVO monitor = algorithmService.getMonitorData(id);
        return Result.success(monitor);
    }

    @Operation(summary = "获取算法统计报表")
    @GetMapping("/{id}/monitor/stats")
    public Result<List<Map<String, Object>>> getMonitorStats(
            @Parameter(description = "算法ID") @PathVariable Long id,
            @Parameter(description = "统计天数，默认7天") @RequestParam(defaultValue = "7") Integer days) {
        List<Map<String, Object>> stats = algorithmService.getMonitorStats(id, days);
        return Result.success(stats);
    }

    // ==================== 导入导出（路径与算法 CRUD 统一为 /algorithms 复数） ====================

    private static final String MODULE = "algorithm";

    @Operation(summary = "导出算法元数据（GET，同步返回文件流或异步返回任务）")
    @GetMapping("/_export")
    @PreAuthorize("@ss.hasPerm('sys:algorithm:export')")
    public Object export(ExportRequest req, HttpServletRequest httpRequest, HttpServletResponse response) {
        Map<String, Object> queryParams = req.toQueryParams(httpRequest.getParameterMap());
        Object result = importExportService.export(MODULE, queryParams, req.getFormat(),
                req.getAsync(), req.getFieldList(), response);
        return result != null ? Result.success(result) : null;
    }

    @Operation(summary = "导出算法元数据（POST，复杂查询条件）")
    @PostMapping("/_export")
    @PreAuthorize("@ss.hasPerm('sys:algorithm:export')")
    public Object exportPost(@RequestBody com.pei.dehaze.model.form.ExportPostRequest req, HttpServletResponse response) {
        Object result = importExportService.export(MODULE, req.getQueryParams(), req.getFormat(),
                req.getAsync(), req.getFields(), response);
        return result != null ? Result.success(result) : null;
    }

    @Operation(summary = "导入算法元数据")
    @PostMapping("/_import")
    @PreAuthorize("@ss.hasPerm('sys:algorithm:import')")
    public Result<Object> importData(ImportRequest req,
                                     @RequestParam("file") MultipartFile file,
                                     HttpServletRequest httpRequest) {
        Map<String, Object> extraParams = req.toExtraParams(httpRequest.getParameterMap());
        Object result = importExportService.importData(MODULE, file, req.getModeOrDefault(),
                req.getAsync(), extraParams);
        return Result.success(result);
    }

    @Operation(summary = "下载算法导入模板")
    @GetMapping("/template")
    @PreAuthorize("@ss.hasPerm('sys:algorithm:import')")
    public void downloadTemplate(@Parameter(description = "文件格式：excel(默认) / csv")
                                 @RequestParam(defaultValue = "excel") String format,
                                 HttpServletResponse response) {
        templateManager.downloadTemplate(MODULE, format, response);
    }
}
