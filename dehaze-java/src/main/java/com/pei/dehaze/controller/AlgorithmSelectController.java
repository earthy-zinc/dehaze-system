package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AlgorithmCompareForm;
import com.pei.dehaze.model.form.AlgorithmTestForm;
import com.pei.dehaze.model.vo.AlgorithmCompareVO;
import com.pei.dehaze.model.vo.AlgorithmDetailVO;
import com.pei.dehaze.model.vo.AlgorithmSelectNodeVO;
import com.pei.dehaze.model.vo.PredictionResultVO;
import com.pei.dehaze.service.AlgorithmSelectService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Tag(name = "17.算法选择")
@RestController
@RequestMapping("/api/v1/algorithms/select")
@RequiredArgsConstructor
public class AlgorithmSelectController {

    private final AlgorithmSelectService algorithmSelectService;

    @Operation(summary = "获取算法选择树（仅已发布算法）")
    @GetMapping("/tree")
    public Result<List<AlgorithmSelectNodeVO>> getTree() {
        return Result.success(algorithmSelectService.getTree());
    }

    @Operation(summary = "获取算法详情（含样例效果图、评分、使用次数）")
    @GetMapping("/{id}")
    public Result<AlgorithmDetailVO> getDetail(
            @Parameter(description = "算法ID") @PathVariable Long id) {
        return Result.success(algorithmSelectService.getDetail(id));
    }

    @Operation(summary = "上传自定义图片测试算法效果")
    @PostMapping("/{id}/test")
    public Result<PredictionResultVO> test(
            @Parameter(description = "算法ID") @PathVariable Long id,
            @Valid @RequestBody AlgorithmTestForm form) {
        return Result.success(algorithmSelectService.test(id, form));
    }

    @Operation(summary = "搜索算法（关键词/拼音/标签）")
    @GetMapping("/search")
    public Result<List<AlgorithmSelectNodeVO>> search(
            @Parameter(description = "搜索关键词") @RequestParam String keyword) {
        return Result.success(algorithmSelectService.search(keyword));
    }

    @Operation(summary = "算法对比（最多3个）")
    @PostMapping("/compare")
    public Result<List<AlgorithmCompareVO>> compare(
            @Valid @RequestBody AlgorithmCompareForm form) {
        return Result.success(algorithmSelectService.compare(form));
    }
}
