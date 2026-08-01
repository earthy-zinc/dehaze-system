package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.PresetForm;
import com.pei.dehaze.model.vo.PresetVO;
import com.pei.dehaze.service.SysPresetService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@Tag(name = "18.去雾处理-参数预设")
@RestController
@RequestMapping("/api/v1/presets")
@RequiredArgsConstructor
public class PresetController {

    private final SysPresetService presetService;

    @Operation(summary = "获取参数预设列表（系统预设 + 用户自定义）")
    @GetMapping
    public PageResult<PresetVO> listPresets(
            @Parameter(description = "算法ID（可选筛选）") @RequestParam(required = false) Long algorithmId,
            @Parameter(description = "页码") @RequestParam(defaultValue = "1") int pageNum,
            @Parameter(description = "每页条数") @RequestParam(defaultValue = "10") int pageSize,
            @Parameter(description = "是否系统预设") @RequestParam(required = false) Boolean isSystem) {
        Page<PresetVO> page = presetService.listPresets(algorithmId, pageNum, pageSize, isSystem);
        return PageResult.success(page);
    }

    @Operation(summary = "创建自定义预设")
    @PostMapping
    public Result<PresetVO> createPreset(@Valid @RequestBody PresetForm form) {
        return Result.success(presetService.createPreset(form));
    }

    @Operation(summary = "更新自定义预设")
    @PutMapping("/{id}")
    public Result<PresetVO> updatePreset(@Parameter(description = "预设ID") @PathVariable Long id,
                                         @Valid @RequestBody PresetForm form) {
        return Result.success(presetService.updatePreset(id, form));
    }

    @Operation(summary = "删除自定义预设")
    @DeleteMapping("/{id}")
    public Result<Void> deletePreset(@Parameter(description = "预设ID") @PathVariable Long id) {
        presetService.deletePreset(id);
        return Result.success();
    }
}
