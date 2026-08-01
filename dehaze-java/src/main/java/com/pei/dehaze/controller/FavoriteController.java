package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.FavoriteForm;
import com.pei.dehaze.model.query.FavoritePageQuery;
import com.pei.dehaze.model.vo.FavoriteCountVO;
import com.pei.dehaze.model.vo.FavoriteStatusVO;
import com.pei.dehaze.model.vo.FavoriteVO;
import com.pei.dehaze.service.FavoriteService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.*;

import java.util.Arrays;
import java.util.List;

@Tag(name = "15.收藏管理")
@RestController
@RequestMapping("/api/v1/favorites")
@RequiredArgsConstructor
public class FavoriteController {

    private final FavoriteService favoriteService;

    @Operation(summary = "收藏列表分页查询")
    @GetMapping("/page")
    public PageResult<FavoriteVO> getPage(@ParameterObject FavoritePageQuery query) {
        Page<FavoriteVO> page = favoriteService.getPage(query);
        return PageResult.success(page);
    }

    @Operation(summary = "添加收藏")
    @PostMapping
    public Result<Long> add(@Valid @RequestBody FavoriteForm form) {
        return Result.success(favoriteService.add(form));
    }

    @Operation(summary = "批量取消收藏")
    @DeleteMapping("/{ids}")
    public Result<Void> deleteByIds(@Parameter(description = "收藏记录ID列表，逗号分隔") @PathVariable String ids) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .map(Long::parseLong)
                .toList();
        favoriteService.deleteByIds(idList);
        return Result.success();
    }

    @Operation(summary = "检查指定对象是否已收藏")
    @GetMapping("/{targetId}/status")
    public Result<FavoriteStatusVO> getStatus(
            @Parameter(description = "收藏对象ID") @PathVariable Long targetId,
            @Parameter(description = "收藏对象类型") @RequestParam String targetType) {
        return Result.success(favoriteService.getStatus(targetType, targetId));
    }

    @Operation(summary = "收藏数量统计")
    @GetMapping("/count")
    public Result<List<FavoriteCountVO>> getCount(
            @Parameter(description = "收藏对象类型（可选）") @RequestParam(required = false) String targetType) {
        return Result.success(favoriteService.getCount(targetType));
    }
}
