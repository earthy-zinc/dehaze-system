package com.pei.dehaze.service.impl;

import cn.hutool.core.lang.Assert;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.util.FileUploadUtils;
import com.pei.dehaze.common.util.ImageUtils;
import com.pei.dehaze.mapper.SysItemFileMapper;
import com.pei.dehaze.model.bo.ItemFileBO;
import com.pei.dehaze.model.dto.ImageFileInfo;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.form.BatchDatasetItemUploadForm;
import com.pei.dehaze.model.form.ImageItemForm;
import com.pei.dehaze.model.form.DatasetItemUploadForm;
import com.pei.dehaze.model.vo.*;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysDatasetService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysItemFileService;
import jakarta.annotation.Resource;
import org.jetbrains.annotations.NotNull;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static com.pei.dehaze.common.util.FileUploadUtils.validateImageFile;

@Service
public class SysItemFileServiceImpl extends ServiceImpl<SysItemFileMapper, SysItemFile>
        implements SysItemFileService {

    @Resource
    private SysFileService sysFileService;

    @Resource
    @Lazy
    private SysDatasetItemService sysDatasetItemService;

    @Resource
    @Lazy
    private SysDatasetService sysDatasetService;

    @Override
    public ImageFileInfo saveItemFile(Long itemId, ItemFileBO itemBO) {
        // 校验图片文件
        validateImageFile(itemBO.getFile());

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

        return ImageFileInfo.builder()
                .id(sysItemFile.getId())
                .datasetItemId(sysItemFile.getItemId())
                .fileId(sysItemFile.getFileId())
                .type(sysItemFile.getType())
                .description(sysItemFile.getDescription())
                .url(sysFile.getUrl())
                .build();
    }


    @Override
    public List<ImageUrlVO> getImageUrlVOs(Long itemId) {
        return this.baseMapper.listImageUrl(itemId);
    }

    public Map<Long, List<ImageUrlVO>> getImageUrlVOsMap(List<Long> itemIds) {
        if (itemIds == null || itemIds.isEmpty()) {
            return new HashMap<>();
        }

        List<ImageUrlVO> allImages = this.baseMapper.listImageUrlByItemIds(itemIds);

        Map<Long, List<ImageUrlVO>> result = new HashMap<>();
        for (ImageUrlVO image : allImages) {
            Long itemId = image.getItemId();
            if (!result.containsKey(itemId)) {
                result.put(itemId, new ArrayList<>());
            }
            result.get(itemId).add(image);
        }

        return result;
    }

    @Override
    public boolean deleteItemFile(Long itemId) {
        SysItemFile sysItemFile = this.getById(itemId);
        Assert.notNull(sysItemFile, "未查询到对应数据项");

        Long fileId = sysItemFile.getFileId();
        Long thumbnailFileId = sysItemFile.getThumbnailFileId();

        boolean res1 = sysFileService.deleteFile(fileId);
        Assert.isTrue(res1, "删除原图失败");

        boolean res2 = sysFileService.deleteFile(thumbnailFileId);
        Assert.isTrue(res2, "删除缩略图失败");

        return this.removeById(itemId);
    }

    @NotNull
    private static ItemFileBO getThumbnailItemBO(ItemFileBO itemBO) {
        File thumbnailFile = ImageUtils.generateThumbnail(itemBO.getFile(), 400, 400);

        try (InputStream thumbnailInputStream = new FileInputStream(thumbnailFile)) {
            long size = thumbnailFile.length();
            String md5 = FileUploadUtils.getMd5(thumbnailInputStream);
            String extension = itemBO.getExtension();
            String name = addSuffix(itemBO.getName(), "_thumbnail");
            String originPath = Paths.get(itemBO.getPath()).getParent().toString();
            String objectName = Path.of("thumbnail", originPath, md5 + "." + extension).toString().replace("\\", "/");
            String path = Path.of("thumbnail", objectName).toString();

            ItemFileBO thumbnailItemBO = new ItemFileBO();
            thumbnailItemBO.setFile(thumbnailFile);
            thumbnailItemBO.setName(name);
            thumbnailItemBO.setObjectName(objectName);
            thumbnailItemBO.setExtension(extension);
            thumbnailItemBO.setMd5(md5);
            thumbnailItemBO.setPath(path);
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
    public ImageDetailVO getImageDetail(Long id) {
        SysItemFile itemFile = this.getById(id);
        Assert.notNull(itemFile, "图片不存在");

        ImageDetailVO detail = new ImageDetailVO();
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

        // 设置分辨率
        if (itemFile.getWidth() != null && itemFile.getHeight() != null) {
            detail.setResolution(itemFile.getWidth() + "x" + itemFile.getHeight());
        }

        // 获取文件信息
        SysFile sysFile = sysFileService.getById(itemFile.getFileId());
        if (sysFile != null) {
            detail.setFileName(sysFile.getName());
            detail.setFileSize(sysFile.getSize());
            detail.setFileFormat(sysFile.getType());
            detail.setUrl(sysFile.getUrl());
        }

        // 获取缩略图URL
        if (itemFile.getThumbnailFileId() != null) {
            SysFile thumbnailFile = sysFileService.getById(itemFile.getThumbnailFileId());
            if (thumbnailFile != null) {
                detail.setThumbnailUrl(thumbnailFile.getUrl());
            }
        }

        // 获取数据项和数据集信息
        SysDatasetItem datasetItem = sysDatasetItemService.getById(itemFile.getItemId());
        if (datasetItem != null) {
            detail.setDatasetId(datasetItem.getDatasetId());
            SysDataset dataset = sysDatasetService.getDatasetById(datasetItem.getDatasetId());
            if (dataset != null) {
                detail.setDatasetName(dataset.getName());
            }
        }

        // 查询同一数据项下的其他图片数量，填充配对信息
        long itemCount = this.count(
            new LambdaQueryWrapper<SysItemFile>()
                .eq(SysItemFile::getItemId, itemFile.getItemId())
        );

        // 设置配对信息
        detail.setHasPairedImages(itemCount > 1);
        detail.setPairedCount((int) itemCount);

        return detail;
    }

    @Override
    public boolean updateImageItemInfo(ImageItemForm form) {
        SysItemFile itemFile = this.getById(form.getItemFileId());
        Assert.notNull(itemFile, "图片不存在");

        // 更新图片类型
        if (form.getType() != null) {
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
    public DatasetItemVO createDatasetItemAndUpload(DatasetItemUploadForm form) {
        // 校验配对图片分辨率一致性
        validatePairedImageResolution(form);

        // 创建数据项
        SysDatasetItem datasetItem = new SysDatasetItem();
        datasetItem.setDatasetId(form.getDatasetId());
        datasetItem.setName(form.getItemName());
        sysDatasetItemService.save(datasetItem);

        // 保存清晰图
        ItemFileBO clearItemBO = FileUploadUtils.createItemFileBO(
                form.getClearImage(),
                "",
                "",
                "clear",
                "",
                form.getSceneType(),
                ""
        );
        ImageFileInfo clearImageInfo = this.saveItemFile(datasetItem.getId(), clearItemBO);

        // 保存有雾图
        List<ItemFileBO> hazyImages = new ArrayList<>();
        for (int i = 0; i < form.getHazyImages().size(); i++) {
            String hazeLevel = form.getHazeLevels().get(i);
            ItemFileBO hazyItemBO = FileUploadUtils.createItemFileBO(
                    form.getHazyImages().get(i),
                    "",
                    "",
                    "hazy",
                    "",
                    form.getSceneType(),
                    hazeLevel
            );
            ImageFileInfo hazyImageInfo = this.saveItemFile(datasetItem.getId(), hazyItemBO);
            hazyImages.add(hazyItemBO);
        }

        // 构建返回结果
        DatasetItemVO result = new DatasetItemVO();
        result.setId(datasetItem.getId());

        return result;
    }

    /**
     * 校验配对图片分辨率一致性
     */
    private void validatePairedImageResolution(DatasetItemUploadForm form) {
        // 校验清晰图
        validateImageFile(form.getClearImage());

        // 解析清晰图宽高
        int[] clearDimensions = FileUploadUtils.getImageDimensions(form.getClearImage());
        if (clearDimensions[0] == 0 || clearDimensions[1] == 0) {
            throw new BusinessException("无法解析清晰图分辨率");
        }

        // 检查每张有雾图的分辨率和格式
        for (int i = 0; i < form.getHazyImages().size(); i++) {
            MultipartFile hazyImage = form.getHazyImages().get(i);

            // 校验文件格式和大小
            validateImageFile(hazyImage);

            // 校验分辨率
            int[] hazyDimensions = FileUploadUtils.getImageDimensions(hazyImage);
            if (hazyDimensions[0] == 0 || hazyDimensions[1] == 0) {
                throw new BusinessException("无法解析有雾图分辨率：" + hazyImage.getOriginalFilename());
            }

            if (hazyDimensions[0] != clearDimensions[0] || hazyDimensions[1] != clearDimensions[1]) {
                throw new BusinessException(String.format(
                    "配对图片分辨率不一致，清晰图：%dx%d，有雾图%s：%dx%d",
                    clearDimensions[0], clearDimensions[1],
                    hazyImage.getOriginalFilename(),
                    hazyDimensions[0], hazyDimensions[1]
                ));
            }
        }
    }

    @Override
    public BatchUploadResultVO batchCreateDatasetItemAndUpload(BatchDatasetItemUploadForm form) {
        int totalGroups = 0;
        int successGroups = 0;
        int failedGroups = 0;
        List<BatchActionFailureDetailVO> failureDetails = new ArrayList<>();

        // 按文件名前缀分组
        Map<String, Map<String, Object>> fileGroups = new HashMap<>();

        for (MultipartFile file : form.getFiles()) {
            String fileName = file.getOriginalFilename();
            if (fileName == null) continue;

            // 提取文件名前缀（去掉下划线和后缀之前的部分）
            String prefix = extractFilePrefix(fileName);

            if (!fileGroups.containsKey(prefix)) {
                fileGroups.put(prefix, new HashMap<>());
                totalGroups++;
            }

            Map<String, Object> group = fileGroups.get(prefix);

            // 判断文件类型
            if (isClearImage(fileName)) {
                group.put("clear", file);
            } else if (isHazyImage(fileName)) {
                String hazeLevel = extractHazeLevel(fileName);
                if (!group.containsKey("hazy")) {
                    group.put("hazy", new ArrayList<Map<String, Object>>());
                }

                Map<String, Object> hazyInfo = new HashMap<>();
                hazyInfo.put("file", file);
                hazyInfo.put("hazeLevel", hazeLevel);
                ((List<Map<String, Object>>) group.get("hazy")).add(hazyInfo);
            }
        }

        // 处理每个分组
        for (Map.Entry<String, Map<String, Object>> entry : fileGroups.entrySet()) {
            String groupName = entry.getKey();
            Map<String, Object> group = entry.getValue();

            try {
                // 验证组完整性
                if (!group.containsKey("clear") || !group.containsKey("hazy")) {
                    throw new BusinessException("配对不完整：缺少清晰图或有雾图");
                }

                // 创建配对上传表单
                DatasetItemUploadForm pairForm = new DatasetItemUploadForm();
                pairForm.setDatasetId(form.getDatasetId());
                pairForm.setItemName(groupName);
                pairForm.setSceneType(form.getSceneType());

                MultipartFile clearImage =
                    (MultipartFile) group.get("clear");
                pairForm.setClearImage(clearImage);

                List<Map<String, Object>> hazyInfos =
                    (List<Map<String, Object>>) group.get("hazy");
                List<MultipartFile> hazyImages = new ArrayList<>();
                List<String> hazeLevels = new ArrayList<>();

                for (Map<String, Object> hazyInfo : hazyInfos) {
                    hazyImages.add((MultipartFile) hazyInfo.get("file"));
                    hazeLevels.add((String) hazyInfo.get("hazeLevel"));
                }

                pairForm.setHazyImages(hazyImages);
                pairForm.setHazeLevels(hazeLevels);

                // 校验分辨率一致性（这会在saveImagePair中自动执行）
                // 保存配对图片
                DatasetItemVO result = this.createDatasetItemAndUpload(pairForm);
                successGroups++;
            } catch (Exception e) {
                BatchActionFailureDetailVO failureDetail = new BatchActionFailureDetailVO();
                failureDetail.setIdentifier(groupName);
                failureDetail.setReason(e.getMessage());
                failureDetails.add(failureDetail);
                failedGroups++;
            }
        }

        BatchUploadResultVO result = new BatchUploadResultVO();
        result.setSuccessCount(successGroups);
        result.setFailedCount(failedGroups);
        result.setTotalFiles(form.getFiles().size());
        result.setMessage(String.format("批量上传完成：成功%d组，失败%d组", successGroups, failedGroups));
        result.setFailureDetails(failureDetails.isEmpty() ? null : failureDetails);
        return result;
    }

    /**
     * 从文件名提取前缀（分组用）
     */
    private String extractFilePrefix(String fileName) {
        // 移除文件扩展名
        String nameWithoutExt = fileName.substring(0, fileName.lastIndexOf('.'));

        // 按下划线分割
        String[] parts = nameWithoutExt.split("_");

        // 返回第一部分作为前缀
        return parts.length > 0 ? parts[0] : nameWithoutExt;
    }

    /**
     * 判断是否为清晰图
     */
    private boolean isClearImage(String fileName) {
        return fileName.contains("_clear") || fileName.contains("_gt");
    }

    /**
     * 判断是否为有雾图
     */
    private boolean isHazyImage(String fileName) {
        return fileName.contains("_hazy");
    }

    /**
     * 从文件名提取雾霾程度
     */
    private String extractHazeLevel(String fileName) {
        // 匹配 *_hazy_light.*, *_hazy_medium.*, *_hazy_heavy.*
        Pattern pattern = Pattern.compile(".*_hazy_(light|medium|heavy).*");
        Matcher matcher = pattern.matcher(fileName);

        if (matcher.matches()) {
            return matcher.group(1);
        }

        // 默认返回medium
        return "medium";
    }
}
