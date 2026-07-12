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
    @Transactional(rollbackFor = Exception.class)
    public ImageUrlVO saveItemFile(Long itemId, ItemFileBO itemBO) {
        // 使用 ImageProcessingService 校验图片文件
        imageProcessingService.validateImageFile(itemBO.getFile());

        // 保存源文件
        SysFile sysFile = sysFileService.saveFile(itemBO);
        // 生成缩略图并保存
        ItemFileBO thumbnailItemBO = getThumbnailItemBO(itemBO);
        SysFile thumbnailSysFile = sysFileService.saveFile(thumbnailItemBO);

        // 查询是否已经存在关联关系
        SysItemFile sysItemFile = this.getOne(new LambdaQueryWrapper<SysItemFile>()
                .eq(SysItemFile::getFileId, sysFile.getId())
                .eq(SysItemFile::getThumbnailFileId, thumbnailSysFile.getId()));

        if (sysItemFile == null) {
            // 保存数据项与文件关联关系
            sysItemFile = new SysItemFile();
            sysItemFile.setItemId(itemId);
            sysItemFile.setFileId(sysFile.getId());
            sysItemFile.setThumbnailFileId(thumbnailSysFile.getId());
            sysItemFile.setType(itemBO.getType());
            sysItemFile.setDescription(itemBO.getDescription());
            // 保存图片宽高
            sysItemFile.setWidth(itemBO.getWidth());
            sysItemFile.setHeight(itemBO.getHeight());
            // 保存上传时的可选标注信息
            sysItemFile.setSceneType(itemBO.getSceneType());
            sysItemFile.setHazeLevel(itemBO.getHazeLevel());
            sysItemFile.setUsageCount(0L);
            this.save(sysItemFile);
        }

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

        // 转换配对图片列表
        if (!pairedFiles.isEmpty()) {
            List<SimpleImageUrlVO> pairedVOList = new ArrayList<>();
            for (SysItemFile pairedFile : pairedFiles) {
                SimpleImageUrlVO simpleVO = new SimpleImageUrlVO();
                simpleVO.setId(pairedFile.getId());
                simpleVO.setItemId(pairedFile.getItemId());
                simpleVO.setDatasetId(detail.getDatasetId());
                simpleVO.setType(pairedFile.getType());

                // 获取文件信息
                SysFile pairedSysFile = sysFileService.getById(pairedFile.getFileId());
                if (pairedSysFile != null) {
                    simpleVO.setUrl(pairedSysFile.getUrl());
                    simpleVO.setFileName(pairedSysFile.getName());
                    simpleVO.setFormattedSize(pairedSysFile.getSize());
                    simpleVO.setFormat(pairedSysFile.getType());
                }

                // 获取缩略图URL
                if (pairedFile.getThumbnailFileId() != null) {
                    SysFile pairedThumbnail = sysFileService.getById(pairedFile.getThumbnailFileId());
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
    public ImageUrlVO convertToImageUrlVO(SysItemFile itemFile) {
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

        // 获取文件信息
        SysFile sysFile = sysFileService.getById(itemFile.getFileId());
        if (sysFile != null) {
            vo.setFileName(sysFile.getName());
            vo.setFormattedSize(sysFile.getSize());
            vo.setFormat(sysFile.getType());
            vo.setUrl(sysFile.getUrl());
            vo.setMd5(sysFile.getMd5());
        }

        // 获取缩略图URL
        if (itemFile.getThumbnailFileId() != null) {
            SysFile thumbnailFile = sysFileService.getById(itemFile.getThumbnailFileId());
            if (thumbnailFile != null) {
                vo.setThumbnailUrl(thumbnailFile.getUrl());
            }
        }

        return vo;
    }
}
