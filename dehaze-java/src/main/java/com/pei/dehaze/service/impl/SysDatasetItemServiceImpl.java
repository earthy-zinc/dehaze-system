package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.mapper.SysDatasetItemMapper;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.query.DatasetItemQuery;
import com.pei.dehaze.model.vo.*;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysDatasetService;
import com.pei.dehaze.service.SysItemFileService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Slf4j
@Service
public class SysDatasetItemServiceImpl extends ServiceImpl<SysDatasetItemMapper, SysDatasetItem>
        implements SysDatasetItemService {

    @Resource
    private SysItemFileService sysItemFileService;

    @Resource
    private SysDatasetService sysDatasetService;

    @Override
    public SysDatasetItem createDatasetItem(Long datasetId) {
        SysDatasetItem datasetItem = new SysDatasetItem();
        datasetItem.setDatasetId(datasetId);
        this.save(datasetItem);
        return datasetItem;
    }

    @Override
    public SysDatasetItem createDatasetItem(Long datasetId, String itemName) {
        SysDatasetItem datasetItem = new SysDatasetItem();
        datasetItem.setDatasetId(datasetId);
        datasetItem.setName(itemName);
        this.save(datasetItem);
        return datasetItem;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteDatasetItem(Long datasetItemId) {
        List<SysItemFile> list = sysItemFileService.list(new LambdaQueryWrapper<SysItemFile>().eq(SysItemFile::getItemId, datasetItemId));
        list.stream().map(SysItemFile::getId).forEach(id -> sysItemFileService.deleteItemFile(id));
        this.removeById(datasetItemId);
    }

    @Override
    public void updateDatasetItem(Long datasetItemId, String itemName) {
        SysDatasetItem datasetItem = this.getById(datasetItemId);
        datasetItem.setName(itemName);
        this.updateById(datasetItem);
    }

    @Override
    public Page<DatasetItemVO> pageSearchDatasetItems(DatasetItemQuery query) {
        // 获取数据集叶子节点ID
        List<Long> leafIds;
        if (query.getDatasetId() != null) {
            leafIds = sysDatasetService.getLeafDatasetId(query.getDatasetId());
        } else {
            leafIds = sysDatasetService.getLeafDatasetIds();
        }

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
                query.getKeywords(),
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
        List<Long> itemIds = results.stream().map(DatasetItemVO::getId).toList();

        page.setRecords(results);
        return page;
    }

    @Override
    public BatchOperationResultVO batchDeleteDatasetItems(List<Long> itemIds) {
        int successCount = 0;
        int failedCount = 0;
        List<BatchActionFailureDetailVO> failureDetails = new ArrayList<>();

        if (itemIds == null || itemIds.isEmpty()) {
            return BatchOperationResultVO.builder()
                    .successCount(0)
                    .failedCount(0)
                    .message("没有需要删除的数据项")
                    .build();
        }

        for (Long itemId : itemIds) {
            try {
                deleteDatasetItem(itemId);
                successCount++;
            } catch (Exception e) {
                log.error("批量删除数据项失败: itemId={}", itemId, e);
                BatchActionFailureDetailVO failureDetail = new BatchActionFailureDetailVO();
                failureDetail.setIdentifier(String.valueOf(itemId));
                failureDetail.setReason(e.getMessage());
                failureDetails.add(failureDetail);
                failedCount++;
            }
        }

        return BatchOperationResultVO.builder()
                .successCount(successCount)
                .failedCount(failedCount)
                .message(String.format("批量删除完成：成功%d个，失败%d个", successCount, failedCount))
                .failureDetails(failureDetails.isEmpty() ? null : failureDetails)
                .build();
    }

    @Override
    public DatasetItemVO getDatasetItem(Long id) {
        // 查询数据项基本信息
        SysDatasetItem datasetItem = this.getById(id);
        if (datasetItem == null) {
            throw new RuntimeException("数据项不存在");
        }

        // 获取数据项下的所有图片
        List<ImageUrlVO> imageUrlVOs = sysItemFileService.getImageUrlVOs(id);

        // 按类型分离图片
        ImageUrlVO clearImage = null;
        List<ImageUrlVO> hazyImages = new ArrayList<>();

        for (ImageUrlVO image : imageUrlVOs) {
            if ("clear".equals(image.getType())) {
                clearImage = image;
            } else if ("hazy".equals(image.getType())) {
                hazyImages.add(image);
            }
        }

        // 按雾霾程度排序有雾图
        hazyImages.sort((a, b) -> {
            if (a.getHazeLevel() == null && b.getHazeLevel() == null) return 0;
            if (a.getHazeLevel() == null) return 1;
            if (b.getHazeLevel() == null) return -1;
            return a.getHazeLevel().compareTo(b.getHazeLevel());
        });

        // 组装返回结果
        DatasetItemVO detail = new DatasetItemVO();
        detail.setId(datasetItem.getId());
        detail.setDatasetId(datasetItem.getDatasetId());
        detail.setName(datasetItem.getName());
        detail.setImageCount(imageUrlVOs.size());
        detail.setClearImage(clearImage);
        detail.setHazyImages(hazyImages);
        detail.setCreateTime(datasetItem.getCreateTime());

        // 如果清晰图有场景类型，设置为数据项的场景类型
        if (clearImage != null && CharSequenceUtil.isNotBlank(clearImage.getSceneType())) {
            detail.setSceneType(clearImage.getSceneType());
        } else if (!hazyImages.isEmpty() && hazyImages.get(0) != null
                && CharSequenceUtil.isNotBlank(hazyImages.get(0).getSceneType())) {
            detail.setSceneType(hazyImages.get(0).getSceneType());
        }

        return detail;
    }
}
