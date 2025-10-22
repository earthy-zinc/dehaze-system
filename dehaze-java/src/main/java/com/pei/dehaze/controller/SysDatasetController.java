package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.base.BasePageQuery;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.converter.DatasetConverter;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.form.DatasetForm;
import com.pei.dehaze.model.query.DatasetQuery;
import com.pei.dehaze.model.vo.DatasetVO;
import com.pei.dehaze.model.vo.ImageItemVO;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysDatasetService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 数据集控制器
 *
 * @author earthyzinc
 * @since 2020/11/6
 */
@Tag(name = "08.数据集接口")
@RestController
@RequestMapping("/api/v1/dataset")
@RequiredArgsConstructor
public class SysDatasetController {

    private final SysDatasetService datasetService;

    private final SysDatasetItemService sysDatasetItemService;

    private final DatasetConverter datasetConverter;
    /**
     * 数据集树形表格
     *
     * @param queryParams 查询参数
     * @return 数据集列表
     */
    @Operation(summary = "获取数据集列表")
    @GetMapping
    public Result<List<DatasetVO>> listDatasets(@ParameterObject DatasetQuery queryParams) {
        List<DatasetVO> datasets = datasetService.getList(queryParams);
        return Result.success(datasets);
    }

    @Operation(summary = "获取数据集下拉选项列表")
    @GetMapping("/options")
    public Result<List<Option<Long>>> getOption() {
        List<Option<Long>> options = datasetService.getOptions();
        return Result.success(options);
    }

    /**
     * 获取数据集信息
     *
     * @param id 数据集id
     * @return 数据集信息
     */
    @Operation(summary = "根据ID获取数据集信息")
    @GetMapping("/{id}")
    public Result<DatasetVO> getDatasetInfoById(@PathVariable Long id) {
        SysDataset sysDataset = datasetService.getDatasetById(id);
        return Result.success(datasetConverter.entity2Vo(sysDataset));
    }

    /**
     * 新增数据集
     *
     * @param dataset 数据集信息
     * @return 操作结果
     */
    @Operation(summary = "新增数据集")
    @PostMapping
    public Result<Void> add(@RequestBody @Valid DatasetForm dataset) {
        boolean result = datasetService.addDataset(dataset);
        return Result.judge(result);
    }

    /**
     * 修改数据集
     *
     * @param id      数据集ID
     * @param dataset 更新后的数据集信息
     * @return 操作结果
     */
    @Operation(summary = "修改数据集")
    @PutMapping("/{id}")
    public Result<Void> update(@PathVariable Long id, @Valid @RequestBody DatasetForm dataset) {
        dataset.setId(id); // 确保ID与路径变量一致
        boolean result = datasetService.updateDataset(dataset);
        return Result.judge(result);
    }

    /**
     * 删除数据集 需要递归删除
     *
     * @param ids 数据集ID数组，字符串形式，例如 "1,2,3"
     * @return 操作结果
     */
    @Operation(summary = "删除数据集")
    @DeleteMapping
    public Result<Void> deleteByIds(@RequestParam List<Long> ids) {
        boolean result = datasetService.deleteDatasets(ids);
        return Result.judge(result);
    }

    /**
     * 获取数据集详细图片
     *
     * @param id 数据集ID
     * @return 图片列表
     */
    @Operation(summary = "获取数据集详细图片")
    @GetMapping("/{id}/images")
    public PageResult<ImageItemVO> getImageItem(@PathVariable Long id, BasePageQuery pageQuery) {
        Page<ImageItemVO> pagedImageItemVOs = sysDatasetItemService.getPagedImageItemVOs(id, pageQuery.getPageNum(), pageQuery.getPageSize());
        return PageResult.success(pagedImageItemVOs);
    }
}
