package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.util.FilePathBuilder;
import com.pei.dehaze.common.util.FileUploadUtils;
import com.pei.dehaze.mapper.SysDatasetItemMapper;
import com.pei.dehaze.mapper.SysDatasetMapper;
import com.pei.dehaze.mapper.SysItemFileMapper;
import com.pei.dehaze.model.bo.ItemFileBO;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.event.ItemFileCreatedEvent;
import com.pei.dehaze.model.event.ItemFileDeletedEvent;
import com.pei.dehaze.model.form.ItemFileUpdateForm;
import com.pei.dehaze.model.vo.BatchDeleteResultVO;
import com.pei.dehaze.model.vo.DatasetItemSimpleVO;
import com.pei.dehaze.model.vo.ImageUrlVO;
import com.pei.dehaze.model.vo.SimpleImageUrlVO;
import com.pei.dehaze.service.ImageProcessingService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysItemFileService;
import jakarta.annotation.Resource;
import org.jetbrains.annotations.NotNull;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class SysItemFileServiceImpl extends ServiceImpl<SysItemFileMapper, SysItemFile>
        implements SysItemFileService {

    @Resource
    private SysFileService sysFileService;

    @Resource
    private ImageProcessingService imageProcessingService;

    @Resource
    private FilePathBuilder filePathBuilder;

    @Resource
    private ApplicationEventPublisher eventPublisher;

    // 使用 Mapper 直接查询，避免循环依赖
    @Resource
    private SysDatasetItemMapper sysDatasetItemMapper;

    @Resource
    private SysDatasetMapper sysDatasetMapper;

    @Override
    public ImageUrlVO saveItemFile(Long itemId, ItemFileBO itemBO) {
        // 1. 图片校验（事务外，避免占用数据库连接）
        imageProcessingService.validateImageFile(itemBO.getFile());

        // 2. 上传源文件到 MinIO（事务外，避免长事务占用连接）
        SysFile sysFile = sysFileService.saveFile(itemBO);
        // 3. 生成缩略图并上传（事务外）
        ItemFileBO thumbnailItemBO = getThumbnailItemBO(itemBO);
        SysFile thumbnailSysFile = sysFileService.saveFile(thumbnailItemBO);

        // 4. 短事务写入 DB 关联记录
        SysItemFile sysItemFile = saveItemFileRecord(itemId, itemBO, sysFile, thumbnailSysFile);

        ImageUrlVO result = new ImageUrlVO();
        result.setId(sysItemFile.getId());
        result.setItemId(sysItemFile.getItemId());
        result.setType(sysItemFile.getType());
        result.setDescription(sysItemFile.getDescription());
        result.setUrl(sysFile.getUrl());
        result.setThumbnailUrl(thumbnailSysFile.getUrl());
        result.setFileName(sysFile.getName());
        result.setFormattedSize(sysFile.getSize());
        result.setFormat(sysFile.getType());
        result.setWidth(sysItemFile.getWidth());
        result.setHeight(sysItemFile.getHeight());
        result.setSceneType(sysItemFile.getSceneType());
        result.setHazeLevel(sysItemFile.getHazeLevel());
        result.setUsageCount(0L);
        result.setCreateTime(sysItemFile.getCreateTime());

        // 发布文件创建事件，通知数据集统计更新
        eventPublisher.publishEvent(new ItemFileCreatedEvent(itemId, sysFile.getId()));

        return result;
    }

    /**
     * 短事务写入数据项文件关联记录
     * 将 DB 写入与 MinIO 上传分离，避免长事务占用数据库连接
     */
    @Transactional(rollbackFor = Exception.class)
    public SysItemFile saveItemFileRecord(Long itemId, ItemFileBO itemBO, SysFile sysFile, SysFile thumbnailSysFile) {
        SysItemFile sysItemFile = new SysItemFile();
        sysItemFile.setItemId(itemId);
        sysItemFile.setFileId(sysFile.getId());
        sysItemFile.setThumbnailFileId(thumbnailSysFile.getId());
        sysItemFile.setType(itemBO.getType());
        sysItemFile.setDescription(itemBO.getDescription());
        sysItemFile.setWidth(itemBO.getWidth());
        sysItemFile.setHeight(itemBO.getHeight());
        sysItemFile.setSceneType(itemBO.getSceneType());
        sysItemFile.setHazeLevel(itemBO.getHazeLevel());
        sysItemFile.setUsageCount(0L);
        this.save(sysItemFile);
        return sysItemFile;
    }


    @Override
    public List<ImageUrlVO> getImageUrlVOs(Long itemId) {
        return this.baseMapper.listImageUrl(itemId);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public boolean deleteFile(Long id) {
        SysItemFile sysItemFile = this.getById(id);
        if (sysItemFile == null) {
            throw new BusinessException("图片不存在");
        }

        // 注：取消配对完整性校验（"必须保留一张清晰图"），适配不同数据集规范

        Long itemId = sysItemFile.getItemId();
        Long fileId = sysItemFile.getFileId();
        Long thumbnailFileId = sysItemFile.getThumbnailFileId();

        boolean res1 = sysFileService.deleteFile(fileId);
        if (!res1) {
            throw new BusinessException("删除原图失败");
        }

        boolean res2 = sysFileService.deleteFile(thumbnailFileId);
        if (!res2) {
            throw new BusinessException("删除缩略图失败");
        }

        boolean result = this.removeById(id);

        // 发布文件删除事件，通知数据集统计更新
        if (result) {
            eventPublisher.publishEvent(new ItemFileDeletedEvent(itemId, fileId));
        }

        return result;
    }

    @Override
    public BatchDeleteResultVO batchDelete(List<Long> ids) {
        BatchDeleteResultVO result = new BatchDeleteResultVO();
        List<Long> successIds = new ArrayList<>();
        List<BatchDeleteResultVO.FailedItem> failedItems = new ArrayList<>();

        for (Long id : ids) {
            try {
                boolean deleted = this.deleteFile(id);
                if (deleted) {
                    successIds.add(id);
                } else {
                    failedItems.add(new BatchDeleteResultVO.FailedItem(id, "删除失败"));
                }
            } catch (Exception e) {
                failedItems.add(new BatchDeleteResultVO.FailedItem(id, e.getMessage()));
            }
        }

        result.setSuccessIds(successIds);
        result.setFailedItems(failedItems);
        result.setSuccessCount(successIds.size());
        result.setFailedCount(failedItems.size());
        return result;
    }

    @NotNull
    private ItemFileBO getThumbnailItemBO(ItemFileBO itemBO) {
        // 使用 ImageProcessingService 生成缩略图
        File thumbnailFile = imageProcessingService.generateThumbnail(itemBO.getFile(), 400, 400);

        try (InputStream thumbnailInputStream = new FileInputStream(thumbnailFile)) {
            long size = thumbnailFile.length();
            String md5 = FileUploadUtils.getMd5(thumbnailInputStream);
            String extension = itemBO.getExtension();
            String name = addSuffix(itemBO.getName(), "_thumbnail");
            // 使用 FilePathBuilder 构建缩略图路径
            String objectName = filePathBuilder.buildThumbnailObjectName(itemBO.getObjectName(), md5, extension);

            ItemFileBO thumbnailItemBO = new ItemFileBO();
            thumbnailItemBO.setFile(thumbnailFile);
            thumbnailItemBO.setName(name);
            thumbnailItemBO.setObjectName(objectName);
            thumbnailItemBO.setExtension(extension);
            thumbnailItemBO.setMd5(md5);
            thumbnailItemBO.setPath(objectName);
            thumbnailItemBO.setSize(size);
            thumbnailItemBO.setDescription(itemBO.getDescription());
            thumbnailItemBO.setType(itemBO.getType());
            return thumbnailItemBO;
        } catch (Exception e) {
            throw new BusinessException("生成缩略图失败", e);
        }
    }

    private static String addSuffix(String fileName, String suffix) {
        // 找到文件名中最后一个"."的位置
        int dotIndex = fileName.lastIndexOf(".");

        // 如果文件名中没有"."，说明没有后缀，直接返回原文件名
        if (dotIndex == -1) {
            return fileName + suffix;
        }

        // 提取文件名部分和扩展名部分
        String namePart = fileName.substring(0, dotIndex); // 文件名
        String extensionPart = fileName.substring(dotIndex); // 文件扩展名（包含点）

        // 在文件名后添加"_thumbnail"并返回结果
        return namePart + suffix + extensionPart;
    }

    @Override
    public ImageUrlVO getImageById(Long id) {
        SysItemFile itemFile = this.getById(id);
        if (itemFile == null) {
            throw new BusinessException("图片不存在");
        }

        ImageUrlVO detail = new ImageUrlVO();
        detail.setId(itemFile.getId());
        detail.setItemId(itemFile.getItemId());
        detail.setType(itemFile.getType());
        detail.setDescription(itemFile.getDescription());
        detail.setSceneType(itemFile.getSceneType());
        detail.setHazeLevel(itemFile.getHazeLevel());
        detail.setWidth(itemFile.getWidth());
        detail.setHeight(itemFile.getHeight());
        detail.setUsageCount(itemFile.getUsageCount() != null ? itemFile.getUsageCount() : 0L);
        detail.setCreateTime(itemFile.getCreateTime());

        // 获取文件信息
        SysFile sysFile = sysFileService.getById(itemFile.getFileId());
        if (sysFile != null) {
            detail.setFileName(sysFile.getName());
            detail.setFormattedSize(sysFile.getSize());
            detail.setFormat(sysFile.getType());
            detail.setUrl(sysFile.getUrl());
            detail.setMd5(sysFile.getMd5());
        }

        // 获取缩略图URL
        if (itemFile.getThumbnailFileId() != null) {
            SysFile thumbnailFile = sysFileService.getById(itemFile.getThumbnailFileId());
            if (thumbnailFile != null) {
                detail.setThumbnailUrl(thumbnailFile.getUrl());
            }
        }

        // 使用 Mapper 直接查询数据项和数据集，避免循环依赖
        SysDatasetItem datasetItem = sysDatasetItemMapper.selectById(itemFile.getItemId());
        if (datasetItem != null) {
            detail.setDatasetId(datasetItem.getDatasetId());

            // 设置数据项简要信息
            DatasetItemSimpleVO datasetItemSimpleVO = new DatasetItemSimpleVO();
            datasetItemSimpleVO.setId(datasetItem.getId());
            datasetItemSimpleVO.setDatasetId(datasetItem.getDatasetId());
            datasetItemSimpleVO.setName(datasetItem.getName());
            detail.setDatasetItem(datasetItemSimpleVO);

            SysDataset dataset = sysDatasetMapper.selectById(datasetItem.getDatasetId());
            if (dataset != null) {
                detail.setDatasetName(dataset.getName());
            }
        }

        // 查询同一数据项下的其他图片，填充配对信息
        List<SysItemFile> pairedFiles = this.list(
            new LambdaQueryWrapper<SysItemFile>()
                .eq(SysItemFile::getItemId, itemFile.getItemId())
                    .ne(SysItemFile::getId, id)
        );

        // 设置配对信息
        detail.setHasPairedImages(!pairedFiles.isEmpty());
        detail.setPairedCount(pairedFiles.size() + 1);

        // 转换配对图片列表（批量查询文件，避免 N+1）
        if (!pairedFiles.isEmpty()) {
            // 批量收集所有 fileId（源文件 + 缩略图）
            Set<Long> allFileIds = new HashSet<>();
            for (SysItemFile pairedFile : pairedFiles) {
                if (pairedFile.getFileId() != null) allFileIds.add(pairedFile.getFileId());
                if (pairedFile.getThumbnailFileId() != null) allFileIds.add(pairedFile.getThumbnailFileId());
            }
            Map<Long, SysFile> fileMap = allFileIds.isEmpty()
                ? Collections.emptyMap()
                : sysFileService.listByIds(allFileIds).stream()
                    .collect(Collectors.toMap(SysFile::getId, f -> f));

            List<SimpleImageUrlVO> pairedVOList = new ArrayList<>();
            for (SysItemFile pairedFile : pairedFiles) {
                SimpleImageUrlVO simpleVO = new SimpleImageUrlVO();
                simpleVO.setId(pairedFile.getId());
                simpleVO.setItemId(pairedFile.getItemId());
                simpleVO.setDatasetId(detail.getDatasetId());
                simpleVO.setType(pairedFile.getType());

                // 从批量查询的 Map 中获取文件信息
                SysFile pairedSysFile = fileMap.get(pairedFile.getFileId());
                if (pairedSysFile != null) {
                    simpleVO.setUrl(pairedSysFile.getUrl());
                    simpleVO.setFileName(pairedSysFile.getName());
                    simpleVO.setFormattedSize(pairedSysFile.getSize());
                    simpleVO.setFormat(pairedSysFile.getType());
                }

                // 从 Map 中获取缩略图URL
                if (pairedFile.getThumbnailFileId() != null) {
                    SysFile pairedThumbnail = fileMap.get(pairedFile.getThumbnailFileId());
                    if (pairedThumbnail != null) {
                        simpleVO.setThumbnailUrl(pairedThumbnail.getUrl());
                    }
                }

                simpleVO.setDescription(pairedFile.getDescription());
                simpleVO.setWidth(pairedFile.getWidth());
                simpleVO.setHeight(pairedFile.getHeight());
                simpleVO.setHazeLevel(pairedFile.getHazeLevel());
                simpleVO.setCreateTime(pairedFile.getCreateTime());

                pairedVOList.add(simpleVO);
            }
            detail.setPairedFiles(pairedVOList);
        }

        return detail;
    }

    @Override
    public boolean updateItemFileInfo(Long id, ItemFileUpdateForm form) {
        SysItemFile itemFile = this.getById(id);
        if (itemFile == null) {
            throw new BusinessException("图片不存在");
        }

        // 更新图片类型（取消配对完整性校验，适配不同数据集规范）
        if (form.getType() != null && !form.getType().equals(itemFile.getType())) {
            itemFile.setType(form.getType());
        }
        // 更新标注信息
        if (form.getSceneType() != null) {
            itemFile.setSceneType(form.getSceneType());
        }
        if (form.getHazeLevel() != null) {
            itemFile.setHazeLevel(form.getHazeLevel());
        }
        if (form.getDescription() != null) {
            itemFile.setDescription(form.getDescription());
        }
        return this.updateById(itemFile);
    }

    @Override
    public void incrementUsageCount(Long id) {
        this.baseMapper.incrementUsageCount(id);
    }

    @Override
    public Map<Long, SysFile> buildFileMap(List<SysItemFile> itemFiles) {
        if (itemFiles == null || itemFiles.isEmpty()) {
            return Collections.emptyMap();
        }
        // 批量收集所有 fileId（源文件 + 缩略图）
        Set<Long> allFileIds = new HashSet<>();
        for (SysItemFile itemFile : itemFiles) {
            if (itemFile.getFileId() != null) {
                allFileIds.add(itemFile.getFileId());
            }
            if (itemFile.getThumbnailFileId() != null) {
                allFileIds.add(itemFile.getThumbnailFileId());
            }
        }
        if (allFileIds.isEmpty()) {
            return Collections.emptyMap();
        }
        return sysFileService.listByIds(allFileIds).stream()
                .collect(Collectors.toMap(SysFile::getId, f -> f));
    }

    @Override
    public ImageUrlVO convertToImageUrlVO(SysItemFile itemFile, Map<Long, SysFile> fileMap) {
        if (itemFile == null) {
            return null;
        }

        ImageUrlVO vo = new ImageUrlVO();
        vo.setId(itemFile.getId());
        vo.setItemId(itemFile.getItemId());
        vo.setType(itemFile.getType());
        vo.setDescription(itemFile.getDescription());
        vo.setSceneType(itemFile.getSceneType());
        vo.setHazeLevel(itemFile.getHazeLevel());
        vo.setWidth(itemFile.getWidth());
        vo.setHeight(itemFile.getHeight());
        vo.setUsageCount(itemFile.getUsageCount() != null ? itemFile.getUsageCount() : 0L);
        vo.setCreateTime(itemFile.getCreateTime());

        // 从预加载的文件Map中获取文件信息（避免N+1查询）
        SysFile sysFile = fileMap.get(itemFile.getFileId());
        if (sysFile != null) {
            vo.setFileName(sysFile.getName());
            vo.setFormattedSize(sysFile.getSize());
            vo.setFormat(sysFile.getType());
            vo.setUrl(sysFile.getUrl());
            vo.setMd5(sysFile.getMd5());
        }

        // 从Map中获取缩略图URL
        if (itemFile.getThumbnailFileId() != null) {
            SysFile thumbnailFile = fileMap.get(itemFile.getThumbnailFileId());
            if (thumbnailFile != null) {
                vo.setThumbnailUrl(thumbnailFile.getUrl());
            }
        }

        return vo;
    }
}
