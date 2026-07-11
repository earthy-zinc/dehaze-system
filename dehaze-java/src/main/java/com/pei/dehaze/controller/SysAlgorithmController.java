package com.pei.dehaze.controller;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.enums.AlgorithmStatusEnum;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.converter.AlgorithmConverter;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysAlgorithmVersion;
import com.pei.dehaze.model.form.AlgorithmAuditForm;
import com.pei.dehaze.model.form.AlgorithmForm;
import com.pei.dehaze.model.form.AlgorithmVersionForm;
import com.pei.dehaze.model.query.AlgorithmQuery;
import com.pei.dehaze.model.vo.AlgorithmMonitorVO;
import com.pei.dehaze.model.vo.AlgorithmVO;
import com.pei.dehaze.model.vo.AlgorithmVersionVO;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.SysAlgorithmVersionService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.nio.charset.StandardCharsets;
import java.util.List;

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
            @Parameter(description = "目标状态") @RequestParam Integer status) {
        boolean result = algorithmService.updateStatus(id, status);
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

    // ==================== 导入导出 ====================

    @Operation(summary = "导出单个算法（配置JSON）")
    @GetMapping("/{id}/_export")
    @PreAuthorize("@ss.hasPerm('sys:algorithm:export')")
    public ResponseEntity<Resource> exportAlgorithm(
            @Parameter(description = "算法ID") @PathVariable Long id) {
        String json = algorithmService.exportAlgorithmJson(id);
        ByteArrayResource resource = new ByteArrayResource(json.getBytes(StandardCharsets.UTF_8));
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "attachment; filename=algorithm_" + id + ".json")
                .body(resource);
    }

    @Operation(summary = "批量导出算法")
    @PostMapping("/_export")
    @PreAuthorize("@ss.hasPerm('sys:algorithm:export')")
    public ResponseEntity<Resource> batchExport(
            @Parameter(description = "算法ID列表") @RequestBody List<Long> ids) {
        StringBuilder sb = new StringBuilder("[\n");
        for (int i = 0; i < ids.size(); i++) {
            sb.append(algorithmService.exportAlgorithmJson(ids.get(i)));
            if (i < ids.size() - 1) {
                sb.append(",\n");
            }
        }
        sb.append("\n]");
        ByteArrayResource resource = new ByteArrayResource(sb.toString().getBytes(StandardCharsets.UTF_8));
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "attachment; filename=algorithms_batch.json")
                .body(resource);
    }

    @Operation(summary = "校验导入包")
    @PostMapping("/_import/validate")
    @PreAuthorize("@ss.hasPerm('sys:algorithm:import')")
    public Result<String> validateImport(@RequestParam("file") MultipartFile file) {
        if (file.isEmpty()) {
            return Result.failed("导入文件不能为空");
        }

        String originalFilename = file.getOriginalFilename();
        if (originalFilename == null || !originalFilename.toLowerCase().endsWith(".json")) {
            return Result.failed("仅支持 .json 格式的算法导出文件");
        }

        try {
            String json = new String(file.getBytes(), StandardCharsets.UTF_8);
            JSONObject root = JSONUtil.parseObj(json);

            // 校验必填字段
            String name = root.getStr("name");
            if (name == null || name.isBlank()) {
                return Result.failed("导入文件缺少必填字段: name");
            }
            String type = root.getStr("type");
            if (type == null || type.isBlank()) {
                return Result.failed("导入文件缺少必填字段: type");
            }

            return Result.success("校验通过: 算法名称=" + name + ", 类型=" + type);
        } catch (Exception e) {
            return Result.failed("导入文件解析失败: " + e.getMessage());
        }
    }

    @Operation(summary = "导入算法")
    @PostMapping("/_import")
    @PreAuthorize("@ss.hasPerm('sys:algorithm:import')")
    public Result<Void> importAlgorithm(@RequestParam("file") MultipartFile file) {
        String originalFilename = file.getOriginalFilename();
        if (originalFilename == null || !originalFilename.toLowerCase().endsWith(".json")) {
            return Result.failed("仅支持 .json 格式的算法导出文件");
        }

        try {
            String json = new String(file.getBytes(), StandardCharsets.UTF_8);
            JSONObject root = JSONUtil.parseObj(json);

            // 解析并校验必填字段
            String name = root.getStr("name");
            if (name == null || name.isBlank()) {
                return Result.failed("导入失败: 缺少算法名称");
            }

            String type = root.getStr("type");
            String description = root.getStr("description", "");
            String importPath = root.getStr("importPath", "");
            String flops = root.getStr("flops", "");
            String params = root.getStr("params", "");
            String version = root.getStr("version", "0.0.1");

            // 检查名称是否已存在
            if (algorithmService.getAllAlgorithms().stream()
                    .anyMatch(a -> name.equals(a.getName()))) {
                return Result.failed("算法名称 '" + name + "' 已存在");
            }

            // 构建算法表单
            AlgorithmForm form = new AlgorithmForm();
            form.setName(name);
            form.setType(type);
            form.setParentId(0L); // 导入的算法默认为顶级
            form.setDescription(description);
            form.setImportPath(importPath);
            form.setStatus(AlgorithmStatusEnum.DRAFT.getValue()); // 导入后为草稿状态

            algorithmService.addAlgorithm(form);
            return Result.success();
        } catch (Exception e) {
            return Result.failed("导入失败: " + e.getMessage());
        }
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
    public Result<AlgorithmMonitorVO> getMonitorStats(
            @Parameter(description = "算法ID") @PathVariable Long id) {
        return getMonitorData(id);
    }
}
