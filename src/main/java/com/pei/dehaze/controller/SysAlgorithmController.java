package com.pei.dehaze.controller;

import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.query.AlgorithmQuery;
import com.pei.dehaze.model.vo.AlgorithmVO;
import com.pei.dehaze.service.SysAlgorithmService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * @author earthy-zinc
 * @since 2024-06-08 19:05:51
 */
@Tag(name = "09.算法接口")
@RestController
@RequestMapping("/api/v1/algorithm")
@RequiredArgsConstructor
public class SysAlgorithmController {
    private final SysAlgorithmService algorithmService;

    /**
     * 获取算法树形表格
     *
     * @param queryParams 查询参数
     * @return 算法列表
     */
    @Operation(summary = "获取算法树形表格")
    @GetMapping
    public Result<List<AlgorithmVO>> getList(@ParameterObject AlgorithmQuery queryParams) {
        List<AlgorithmVO> algorithms = algorithmService.getList(queryParams);
        return Result.success(algorithms);
    }

    /**
     * 获取模型下拉选项列表
     *
     * @return 模型下拉选项列表
     */
    @Operation(summary = "获取模型下拉选项列表")
    @GetMapping("/options")
    public Result<List<Option<?>>> getOption() {
        List<Option<?>> options = algorithmService.getOption();
        return Result.success(options);
    }

    /**
     * 新增算法
     *
     * @param data 算法对象
     * @return 操作结果
     */
    @Operation(summary = "新增算法")
    @PostMapping
    public Result<Void> add(@RequestBody AlgorithmVO data) {
//        algorithmService.add(data);
        return Result.success(null);
    }

    /**
     * 修改算法
     *
     * @param id   算法ID
     * @param data 算法对象
     * @return 操作结果
     */
    @Operation(summary = "修改算法")
    @PutMapping("/{id}")
    public Result<Void> update(@PathVariable Long id, @RequestBody AlgorithmVO data) {
        data.setId(id); // 确保ID与路径变量一致
//        algorithmService.update(data);
        return Result.success(null);
    }

    /**
     * 删除算法
     *
     * @param ids 算法ID数组，逗号分隔
     * @return 操作结果
     */
    @Operation(summary = "删除算法")
    @DeleteMapping
    public Result<Void> deleteByIds(@RequestParam String ids) {
//        algorithmService.deleteByIds(ids.split(","));
        return Result.success(null);
    }
}
