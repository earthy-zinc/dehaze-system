package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.BatchDeleteRequest;
import com.pei.dehaze.model.form.DatasetAddForm;
import com.pei.dehaze.model.form.DatasetUpdateForm;
import com.pei.dehaze.model.query.DatasetQuery;
import com.pei.dehaze.model.vo.BatchDeleteResult;
import com.pei.dehaze.model.vo.DatasetVO;
import com.pei.dehaze.service.DatasetOperationService;
import com.pei.dehaze.service.SysDatasetService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Tag(name = "08.数据集接口")
@RestController
@RequestMapping("/api/v1/datasets")
@RequiredArgsConstructor
public class SysDatasetController {

    private final SysDatasetService datasetService;

    private final DatasetOperationService datasetOperationService;

    @Operation(
            summary = "分页查询数据集列表",
            description = "分页获取根级数据集列表，支持关键字搜索、类型和状态筛选。返回树形结构，根节点分页，" +
                    "子节点通过懒加载接口获取。包含统计信息（图片数量等）。适用于数据集管理页面。"
    )
    @GetMapping
    public PageResult<DatasetVO> listDatasets(@ParameterObject DatasetQuery queryParams) {
        IPage<DatasetVO> page = datasetService.listPagedDatasets(queryParams);
        return PageResult.success(page);
    }

    @Operation(
            summary = "获取子数据集列表（懒加载）",
            description = "根据父数据集ID获取直接子数据集列表，用于树形表格懒加载。返回子节点列表，" +
                    "每个节点包含hasChildren标记表示是否有下级节点。"
    )
    @GetMapping("/children/{parentId}")
    public Result<List<DatasetVO>> listChildren(
            @Parameter(description = "父数据集ID", required = true, example = "1")
            @PathVariable Long parentId
    ) {
        List<DatasetVO> children = datasetService.listChildren(parentId);
        return Result.success(children);
    }

    @Operation(
            summary = "获取数据集下拉选项列表",
            description = "获取用于下拉选择器的数据集选项列表，返回树形结构的label-value对。" +
                    "只返回启用状态的数据集，按名称排序。适用于前端选择器组件、父数据集选择等场景。"
    )
    @GetMapping("/options")
    public Result<List<Option<Long>>> getOption() {
        List<Option<Long>> options = datasetService.getOptions();
        return Result.success(options);
    }

    @Operation(
            summary = "根据ID获取数据集详细信息",
            description = "根据数据集ID获取完整的数据集信息，包括基本信息、统计数据（图片数量、使用次数）、" +
                    "分布统计（场景类型、雾霾程度、文件格式）和子数据集列表。统计信息通过Redis缓存优化，响应时间<200ms。"
    )
    @GetMapping("/{id}")
    public Result<DatasetVO> getDatasetInfoById(
            @Parameter(description = "数据集ID", required = true, example = "1")
            @PathVariable
            Long id
    ) {
        DatasetVO datasetVO = datasetService.getDatasetById(id);
        return Result.success(datasetVO);
    }

    @Operation(
            summary = "新增数据集",
            description = "创建新的数据集，支持树形结构管理。系统会自动生成数据集存储目录，" +
                    "并验证父数据集存在性和名称唯一性。创建成功后可立即使用。"
    )
    @PostMapping
    public Result<Long> add(@RequestBody @Valid DatasetAddForm dataset) {
        DatasetVO result = datasetService.addDataset(dataset);
        return Result.success(result.getId());
    }

    @Operation(
            summary = "修改数据集信息",
            description = "更新指定数据集的详细信息，支持修改名称、描述、类型和状态。" +
                    "修改名称时会验证唯一性，禁用数据集后将不可用。系统自动更新修改时间。"
    )
    @PutMapping("/{id}")
    public Result<DatasetVO> update(
            @Parameter(description = "数据集ID", required = true, example = "1")
            @PathVariable
            Long id,
            @Valid @RequestBody DatasetUpdateForm dataset
    ) {
        DatasetVO result = datasetService.updateDataset(id, dataset);
        return Result.success(result);
    }

    @Operation(
            summary = "删除单个数据集",
            description = "删除指定的数据集，支持级联删除。删除范围包括：数据集本身、所有子数据集、" +
                    "关联的图片文件、缩略图文件和统计缓存。删除操作不可逆，请谨慎使用。"
    )
    @DeleteMapping("/{id}")
    public Result<Void> deleteDataset(
            @Parameter(description = "数据集ID", required = true, example = "1")
            @PathVariable
            Long id
    ) {
        datasetService.deleteDataset(id);
        return Result.success();
    }

    @Operation(
            summary = "批量删除数据集",
            description = "批量删除指定的数据集，支持级联删除。删除范围包括：数据集本身、所有子数据集、" +
                    "关联的图片文件、缩略图文件和统计缓存。返回每个数据集的删除结果。"
    )
    @DeleteMapping("/batch")
    public Result<BatchDeleteResult> batchDeleteDatasets(
            @Valid @RequestBody BatchDeleteRequest request
    ) {
        BatchDeleteResult result = datasetOperationService.batchDeleteDatasets(request.getIds());
        return Result.success(result);
    }
}
