package com.pei.dehaze.controller;

import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 部门控制器
 *
 * @author haoxr
 * @since 2020/11/6
 */
@Tag(name = "05.数据集接口")
@RestController
@RequestMapping("/api/v1/dataset")
@RequiredArgsConstructor
public class DatasetController {
//    private final DatasetService datasetService;
//
//    @Operation(summary = "获取数据集列表")
//    @GetMapping
//    public Result<?> listDatasets(
//            @ParameterObject DatasetQuery queryParams
//    ){
//        return null;
//    }
}
