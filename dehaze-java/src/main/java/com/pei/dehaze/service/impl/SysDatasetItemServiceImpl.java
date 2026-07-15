package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.util.ImageClassificationUtils;
import com.pei.dehaze.mapper.SysDatasetItemMapper;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.query.DatasetItemQuery;
import com.pei.dehaze.model.vo.*;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysDatasetService;
import com.pei.dehaze.service.SysItemFileService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 数据项服务实现
 * 职责：处理DatasetItem（数据项）的基础CRUD操作
 * 注意：跨服务的复合操作（如级联删除、创建数据项并上传图片）已迁移到 DatasetOperationService
 *
 * @author earthy-zinc
 * @since 2024-06-08
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class SysDatasetItemServiceImpl extends ServiceImpl<SysDatasetItemMapper, SysDatasetItem>
        implements SysDatasetItemService {

    private final SysItemFileService sysItemFileService;

    @Lazy
    private final SysDatasetService sysDatasetService;

    @Override
    public SysDatasetItem createDatasetItem(Long datasetId, String itemName) {
        SysDatasetItem datasetItem = new SysDatasetItem();
        datasetItem.setDatasetId(datasetId);
        datasetItem.setName(itemName);
        this.save(datasetItem);
        return datasetItem;
    }

    @Override
    public DatasetItemVO createAndReturnDatasetItem(Long datasetId, String itemName) {
        SysDatasetItem datasetItem = createDatasetItem(datasetId, itemName);
        return getDatasetItem(datasetItem.getId());
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteDatasetItem(Long datasetItemId) {
        // 注意：此方法仅删除数据项记录本身，不级联删除文件
        // 如需级联删除（包含文件），请使用 DatasetOperationService.deleteDatasetItemCascade()
        this.removeById(datasetItemId);
    }

    @Override
    public void updateDatasetItem(Long datasetItemId, String itemName) {
        SysDatasetItem datasetItem = this.getById(datasetItemId);
        if (datasetItem == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据项不存在");
        }
        datasetItem.setName(itemName);
        this.updateById(datasetItem);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public DatasetItemVO updateAndReturnDatasetItem(Long datasetItemId, String itemName, String sceneType) {
        SysDatasetItem datasetItem = this.getById(datasetItemId);
        if (datasetItem == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据项不存在");
        }

        // 更新数据项名称
        if (itemName != null) {
            datasetItem.setName(itemName);
            this.updateById(datasetItem);
        }

        // 更新场景类型需要更新所有关联的图片
        if (sceneType != null) {
            List<SysItemFile> itemFiles = sysItemFileService.list(
                    new LambdaQueryWrapper<SysItemFile>()
                            .eq(SysItemFile::getItemId, datasetItemId)
            );
            for (SysItemFile itemFile : itemFiles) {
                itemFile.setSceneType(sceneType);
            }
            sysItemFileService.updateBatchById(itemFiles);
        }

        return getDatasetItem(datasetItemId);
    }

    @Override
    public Page<DatasetItemVO> pageSearchDatasetItems(DatasetItemQuery query) {
        // 获取数据集叶子节点ID
        List<Long> leafIds = query.getDatasetId() != null
                ? sysDatasetService.getLeafDatasetId(query.getDatasetId())
                : sysDatasetService.getLeafDatasetIds();

        if (leafIds.isEmpty()) {
            Page<DatasetItemVO> emptyPage = new Page<>();
            emptyPage.setRecords(Collections.emptyList());
            emptyPage.setTotal(0);
            return emptyPage;
        }

        // 使用Mapper执行复杂查询
        Page<DatasetItemVO> page = new Page<>(query.getPageNum(), query.getPageSize());
        List<DatasetItemVO> results = this.baseMapper.searchImages(
                page,
                leafIds,
                query.getKeyword(),
                query.getSceneType(),
                query.getHazeLevel(),
                query.getMinWidth(),
                query.getMaxWidth(),
                query.getMinHeight(),
                query.getMaxHeight(),
                query.getMinSize(),
                query.getMaxSize(),
                query.getSortBy(),
                query.getSortOrder()
        );

        // 批量查询图片信息，避免N+1问题
        if (!results.isEmpty()) {
            List<Long> itemIds = results.stream().map(DatasetItemVO::getId).toList();

            // 批量获取所有数据项的图片
            List<SysItemFile> allItemFiles = sysItemFileService.list(
                    new LambdaQueryWrapper<SysItemFile>()
                            .in(SysItemFile::getItemId, itemIds)
                            .orderByAsc(SysItemFile::getItemId)
                            .orderByAsc(SysItemFile::getType)
            );

            // 批量预加载所有文件信息（源文件 + 缩略图），避免转换VO时N+1查询
            Map<Long, SysFile> fileMap = sysItemFileService.buildFileMap(allItemFiles);

            // 按itemId分组
            Map<Long, List<SysItemFile>> itemFilesMap = allItemFiles.stream()
                    .collect(Collectors.groupingBy(SysItemFile::getItemId));

            // 填充每个DatasetItemVO的图片信息
            for (DatasetItemVO itemVO : results) {
                List<SysItemFile> itemFiles = itemFilesMap.getOrDefault(itemVO.getId(), Collections.emptyList());
                fillDatasetItemImages(itemVO, itemFiles, fileMap);
            }
        }

        page.setRecords(results);
        return page;
    }

    /**
     * 填充数据项的图片信息
     * 使用统一的图片分类工具类
     *
     * @param itemVO    数据项VO
     * @param itemFiles 图片文件列表
     * @param fileMap   预加载的文件Map（fileId -> SysFile）
     */
    private void fillDatasetItemImages(DatasetItemVO itemVO, List<SysItemFile> itemFiles, Map<Long, SysFile> fileMap) {
        log.debug("填充数据项图片信息: itemId={}, 查询到图片数量={}", itemVO.getId(), itemFiles.size());

        // 使用统一的图片分类工具类，通过预加载的fileMap避免N+1查询
        ImageClassificationUtils.ClassificationResult result =
                ImageClassificationUtils.classifyItemFiles(
                        itemFiles,
                        itemFile -> sysItemFileService.convertToImageUrlVO(itemFile, fileMap)
                );

        itemVO.setImageCount(itemFiles.size());
        itemVO.setClearImage(result.getClearImage());
        itemVO.setHazyImages(result.getHazyImages());
        itemVO.setSceneType(result.getSceneType());
    }

    @Override
    public DatasetItemVO getDatasetItem(Long id) {
        // 查询数据项基本信息
        SysDatasetItem datasetItem = this.getById(id);
        if (datasetItem == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据项不存在");
        }

        // 获取数据项下的所有图片
        List<ImageUrlVO> imageUrlVOs = sysItemFileService.getImageUrlVOs(id);

        // 使用统一的图片分类工具类
        ImageClassificationUtils.ClassificationResult result =
                ImageClassificationUtils.classifyImages(imageUrlVOs);

        // 组装返回结果
        DatasetItemVO detail = new DatasetItemVO();
        detail.setId(datasetItem.getId());
        detail.setDatasetId(datasetItem.getDatasetId());
        detail.setName(datasetItem.getName());
        detail.setImageCount(imageUrlVOs.size());
        detail.setClearImage(result.getClearImage());
        detail.setHazyImages(result.getHazyImages());
        detail.setSceneType(result.getSceneType());
        detail.setCreateTime(datasetItem.getCreateTime());
        detail.setUpdateTime(datasetItem.getUpdateTime());

        return detail;
    }
}
