package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.HistoryForm;
import com.pei.dehaze.model.form.HistoryUpdateForm;
import com.pei.dehaze.model.query.HistoryQuery;
import com.pei.dehaze.model.vo.InputHistoryVO;
import com.pei.dehaze.service.SysInputHistoryService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 图像输入模块 —— 历史记录管理
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Tag(name = "12.图像输入历史记录")
@RestController
@RequestMapping("/api/v1/image-input/history")
@RequiredArgsConstructor
public class ImageInputController {

    private final SysInputHistoryService historyService;

    @Operation(summary = "分页查询历史记录")
    @GetMapping
    public PageResult<InputHistoryVO> getHistoryPage(@ParameterObject HistoryQuery query) {
        Page<InputHistoryVO> page = historyService.getHistoryPage(query);
        return PageResult.success(page);
    }

    @Operation(summary = "获取历史记录详情")
    @GetMapping("/{id}")
    public Result<InputHistoryVO> getHistoryById(
            @Parameter(description = "记录ID") @PathVariable Long id) {
        InputHistoryVO vo = historyService.getHistoryById(id);
        return Result.success(vo);
    }

    @Operation(summary = "创建历史记录")
    @PostMapping
    public Result<Long> createHistory(@Valid @RequestBody HistoryForm form) {
        Long id = historyService.createHistory(form);
        return Result.success(id);
    }

    @Operation(summary = "更新历史记录（如添加收藏）")
    @PutMapping("/{id}")
    public Result<Void> updateHistory(
            @Parameter(description = "记录ID") @PathVariable Long id,
            @RequestBody HistoryUpdateForm form) {
        boolean result = historyService.updateHistory(id, form);
        return Result.judge(result);
    }

    @Operation(summary = "删除单条历史记录")
    @DeleteMapping("/{id}")
    public Result<Void> deleteHistory(
            @Parameter(description = "记录ID") @PathVariable Long id) {
        boolean result = historyService.deleteHistory(id);
        return Result.judge(result);
    }

    @Operation(summary = "批量删除历史记录")
    @DeleteMapping("/batch")
    public Result<Integer> batchDeleteHistory(@RequestBody List<Long> ids) {
        int count = historyService.batchDeleteHistory(ids);
        return Result.success(count);
    }

    @Operation(summary = "清空所有历史记录")
    @DeleteMapping("/clear")
    public Result<Integer> clearAllHistory() {
        int count = historyService.clearAllHistory();
        return Result.success(count);
    }

    @Operation(summary = "同步本地与云端历史记录")
    @PostMapping("/sync")
    public Result<Integer> syncHistory() {
        int result = historyService.syncHistory();
        return Result.success(result);
    }
}
