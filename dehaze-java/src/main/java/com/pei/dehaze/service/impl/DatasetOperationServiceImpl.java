package com.pei.dehaze.service.impl;

import cn.hutool.core.lang.Assert;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.util.FileBOFactory;
import com.pei.dehaze.model.bo.ItemFileBO;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.form.BatchDatasetItemUploadForm;
import com.pei.dehaze.model.form.DatasetItemUploadForm;
import com.pei.dehaze.model.vo.BatchActionFailureDetailVO;
import com.pei.dehaze.model.vo.BatchDeleteResult;
import com.pei.dehaze.model.vo.BatchOperationResultVO;
import com.pei.dehaze.model.vo.BatchUploadFailedItemVO;
import com.pei.dehaze.model.vo.BatchUploadResultVO;
import com.pei.dehaze.model.vo.BatchUploadSuccessItemVO;
import com.pei.dehaze.model.vo.DatasetItemVO;
import com.pei.dehaze.service.DatasetOperationService;
import com.pei.dehaze.service.ImageProcessingService;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysDatasetService;
import com.pei.dehaze.service.SysItemFileService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 数据集操作服务实现
 * 处理跨服务的复杂组合操作，避免循环依赖
 *
 * @author earthy-zinc
 * @since 2025-12-13
 */
@Slf4j
@Service
public class DatasetOperationServiceImpl implements DatasetOperationService {

    @Lazy
    @Resource
    private SysDatasetService sysDatasetService;

    @Resource
    private SysDatasetItemService sysDatasetItemService;

    @Resource
    private SysItemFileService sysItemFileService;

    @Resource
    private FileBOFactory fileBOFactory;

    @Resource
    private ImageProcessingService imageProcessingService;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public DatasetItemVO createDatasetItemWithImages(DatasetItemUploadForm form) {
        // 校验配对图片分辨率一致性
        validatePairedImageResolution(form);

        // 创建数据项
        SysDatasetItem datasetItem = sysDatasetItemService.createDatasetItem(form.getDatasetId(), form.getName());

        // 保存清晰图
        ItemFileBO clearItemBO = fileBOFactory.createItemFileBO(
                form.getClearImage(),
                "",
                "clear",
                "",
                form.getSceneType(),
                ""
        );
        sysItemFileService.saveItemFile(datasetItem.getId(), clearItemBO);

        // 保存有雾图
        for (int i = 0; i < form.getHazyImages().size(); i++) {
            String hazeLevel = form.getHazeLevels().get(i);
            ItemFileBO hazyItemBO = fileBOFactory.createItemFileBO(
                    form.getHazyImages().get(i),
                    "",
                    "hazy",
                    "",
                    form.getSceneType(),
                    hazeLevel
            );
            sysItemFileService.saveItemFile(datasetItem.getId(), hazyItemBO);
        }

        // 构建返回结果
        return sysDatasetItemService.getDatasetItem(datasetItem.getId());
    }

    /**
     * 校验配对图片分辨率一致性
     */
    private void validatePairedImageResolution(DatasetItemUploadForm form) {
        // 校验清晰图
        imageProcessingService.validateImageFile(form.getClearImage());

        // 解析清晰图宽高
        int[] clearDimensions = imageProcessingService.getImageDimensions(form.getClearImage());
        if (clearDimensions[0] == 0 || clearDimensions[1] == 0) {
            throw new BusinessException("无法解析清晰图分辨率");
        }

        // 检查每张有雾图的分辨率和格式
        for (int i = 0; i < form.getHazyImages().size(); i++) {
            MultipartFile hazyImage = form.getHazyImages().get(i);

            // 校验文件格式和大小
            imageProcessingService.validateImageFile(hazyImage);

            // 校验分辨率
            int[] hazyDimensions = imageProcessingService.getImageDimensions(hazyImage);
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
    @Transactional(rollbackFor = Exception.class)
    public BatchUploadResultVO batchCreateDatasetItemsWithImages(BatchDatasetItemUploadForm form) {
        int successGroups = 0;
        int failedGroups = 0;
        List<BatchUploadSuccessItemVO> successItems = new ArrayList<>();
        List<BatchUploadFailedItemVO> failedItems = new ArrayList<>();

        // 按文件名前缀分组
        Map<String, Map<String, Object>> fileGroups = new HashMap<>();

        for (MultipartFile file : form.getFiles()) {
            String fileName = file.getOriginalFilename();
            if (fileName == null) continue;

            // 提取文件名前缀（去掉下划线和后缀之前的部分）
            String prefix = extractFilePrefix(fileName);

            if (!fileGroups.containsKey(prefix)) {
                fileGroups.put(prefix, new HashMap<>());
            }

            Map<String, Object> group = fileGroups.get(prefix);

            // 判断文件类型
            if (isClearImage(fileName)) {
                group.put("clear", file);
            } else if (isHazyImage(fileName)) {
                try {
                    String hazeLevel = extractHazeLevel(fileName);
                    if (!group.containsKey("hazy")) {
                        group.put("hazy", new ArrayList<Map<String, Object>>());
                    }

                    Map<String, Object> hazyInfo = new HashMap<>();
                    hazyInfo.put("file", file);
                    hazyInfo.put("hazeLevel", hazeLevel);
                    ((List<Map<String, Object>>) group.get("hazy")).add(hazyInfo);
                } catch (Exception e) {
                    log.warn("解析雾霾程度失败: {}", fileName);
                    // 记录单个文件错误
                    failedItems.add(new BatchUploadFailedItemVO(fileName, "解析雾霾程度失败: " + e.getMessage()));
                }
            }
        }

        // 处理每个分组
        for (Map.Entry<String, Map<String, Object>> entry : fileGroups.entrySet()) {
            String groupName = entry.getKey();
            Map<String, Object> group = entry.getValue();

            try {
                // 验证组完整性
                if (!group.containsKey("clear")) {
                    throw new BusinessException("缺少清晰图（需包含_clear或_gt后缀）");
                }
                if (!group.containsKey("hazy")) {
                    throw new BusinessException("缺少有雾图（需包含_hazy后缀）");
                }

                // 创建配对上传表单
                DatasetItemUploadForm pairForm = new DatasetItemUploadForm();
                pairForm.setDatasetId(form.getDatasetId());
                pairForm.setName(groupName);
                pairForm.setSceneType(form.getSceneType());

                MultipartFile clearImage = (MultipartFile) group.get("clear");
                pairForm.setClearImage(clearImage);

                List<Map<String, Object>> hazyInfos = (List<Map<String, Object>>) group.get("hazy");
                List<MultipartFile> hazyImages = new ArrayList<>();
                List<String> hazeLevels = new ArrayList<>();

                for (Map<String, Object> hazyInfo : hazyInfos) {
                    hazyImages.add((MultipartFile) hazyInfo.get("file"));
                    hazeLevels.add((String) hazyInfo.get("hazeLevel"));
                }

                pairForm.setHazyImages(hazyImages);
                pairForm.setHazeLevels(hazeLevels);

                // 保存配对图片并获取返回结果
                DatasetItemVO createdItem = this.createDatasetItemWithImages(pairForm);
                successGroups++;

                // 记录成功项详情：1张清晰图 + N张有雾图
                int fileCount = 1 + hazyImages.size();
                successItems.add(new BatchUploadSuccessItemVO(createdItem.getId(), groupName, fileCount));

            } catch (Exception e) {
                failedItems.add(new BatchUploadFailedItemVO(groupName, e.getMessage()));
                failedGroups++;
            }
        }

        BatchUploadResultVO result = new BatchUploadResultVO();
        result.setTotal(form.getFiles().size());
        result.setSucceeded(successGroups);
        result.setFailed(failedGroups);
        result.setSuccessItems(successItems);
        result.setFailedItems(failedItems);
        return result;
    }

    /**
     * 从文件名提取前缀（分组用）
     */
    private String extractFilePrefix(String fileName) {
        String nameWithoutExt = fileName.substring(0, fileName.lastIndexOf('.'));
        // 移除 _clear, _gt, _hazy 及其后缀
        // 假设命名规则是 prefix_clear.jpg 或 prefix_hazy_light.jpg
        return nameWithoutExt.replaceAll("(_clear|_gt|_hazy.*)$", "");
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

        // 如果无法从文件名提取，抛出异常，要求文件名必须包含雾霾程度
        throw new BusinessException("文件名必须包含雾霾程度标识(light/medium/heavy)，例如：name_hazy_light.jpg");
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteDatasetItemCascade(Long datasetItemId) {
        Assert.notNull(datasetItemId, "数据项ID不能为空");

        // 检查数据项是否存在
        SysDatasetItem datasetItem = sysDatasetItemService.getById(datasetItemId);
        if (datasetItem == null) {
            throw new BusinessException("数据项不存在");
        }

        // 先删除数据项下的所有图片文件
        List<SysItemFile> itemFiles = sysItemFileService.list(
                new LambdaQueryWrapper<SysItemFile>()
                        .eq(SysItemFile::getItemId, datasetItemId)
        );

        for (SysItemFile itemFile : itemFiles) {
            sysItemFileService.deleteFile(itemFile.getId());
        }

        // 再删除数据项本身
        sysDatasetItemService.removeById(datasetItemId);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void batchDeleteDatasetItemsCascade(List<Long> datasetItemIds) {
        if (datasetItemIds == null || datasetItemIds.isEmpty()) {
            return;
        }

        for (Long datasetItemId : datasetItemIds) {
            try {
                deleteDatasetItemCascade(datasetItemId);
            } catch (Exception e) {
                log.error("删除数据项失败: datasetItemId={}", datasetItemId, e);
                throw new BusinessException("删除数据项失败: " + e.getMessage());
            }
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public BatchOperationResultVO batchDeleteDatasetItemsCascadeWithResult(List<Long> datasetItemIds) {
        int successCount = 0;
        int failedCount = 0;
        List<Long> successIds = new ArrayList<>();
        List<BatchActionFailureDetailVO> failureDetails = new ArrayList<>();

        if (datasetItemIds == null || datasetItemIds.isEmpty()) {
            return BatchOperationResultVO.builder()
                    .successCount(0)
                    .failedCount(0)
                    .message("没有需要删除的数据项")
                    .build();
        }

        for (Long datasetItemId : datasetItemIds) {
            try {
                deleteDatasetItemCascade(datasetItemId);
                successCount++;
                successIds.add(datasetItemId);
            } catch (Exception e) {
                log.error("批量删除数据项失败: datasetItemId={}", datasetItemId, e);
                BatchActionFailureDetailVO failureDetail = new BatchActionFailureDetailVO();
                failureDetail.setIdentifier(String.valueOf(datasetItemId));
                failureDetail.setReason(e.getMessage());
                failureDetails.add(failureDetail);
                failedCount++;
            }
        }

        return BatchOperationResultVO.builder()
                .successCount(successCount)
                .failedCount(failedCount)
                .successIds(successIds)
                .message(String.format("批量删除完成：成功%d个，失败%d个", successCount, failedCount))
                .failureDetails(failureDetails.isEmpty() ? null : failureDetails)
                .build();
    }

    @Override
    public BatchDeleteResult batchDeleteDatasets(List<Long> datasetIds) {
        if (datasetIds == null || datasetIds.isEmpty()) {
            throw new BusinessException("删除的数据集ID列表不能为空");
        }

        int total = datasetIds.size();
        int succeeded = 0;
        int failed = 0;
        List<BatchDeleteResult.DeleteResultItem> results = new ArrayList<>();

        for (Long datasetId : datasetIds) {
            try {
                // 在删除前先获取父数据集ID，用于后续清除祖先缓存
                SysDataset dataset = sysDatasetService.getById(datasetId);
                Long parentId = dataset != null ? dataset.getParentId() : null;

                // 获取该数据集及其所有子数据集的ID
                List<Long> allDatasetIds = sysDatasetService.getDatasetAndDescendantIds(datasetId);
                // 获取所有叶子节点数据集下的数据项
                List<Long> leafDatasetIds = sysDatasetService.getLeafDatasetId(datasetId);

                // 删除所有数据项（包括其下的图片文件）
                for (Long leafId : leafDatasetIds) {
                    List<SysDatasetItem> items = sysDatasetItemService.list(
                            new LambdaQueryWrapper<SysDatasetItem>()
                                    .eq(SysDatasetItem::getDatasetId, leafId)
                    );

                    for (SysDatasetItem item : items) {
                        deleteDatasetItemCascade(item.getId());
                    }
                }

                // 删除数据集本身（从叶子节点往上删）
                allDatasetIds = allDatasetIds.stream().distinct().toList();
                for (int i = allDatasetIds.size() - 1; i >= 0; i--) {
                    sysDatasetService.removeById(allDatasetIds.get(i));
                }

                succeeded++;
                // 清除已删除数据集的缓存
                for (Long deletedId : allDatasetIds) {
                    sysDatasetService.evictDatasetStatsCache(deletedId);
                }
                // 清除父数据集及其祖先的统计缓存（如果父数据集存在且不是根节点）
                if (parentId != null && parentId != 0L) {
                    sysDatasetService.evictDatasetAndAncestorStatsCache(parentId);
                }
                results.add(BatchDeleteResult.DeleteResultItem.builder()
                        .id(datasetId)
                        .status("success")
                        .build());

            } catch (BusinessException e) {
                failed++;
                results.add(BatchDeleteResult.DeleteResultItem.builder()
                        .id(datasetId)
                        .status("failed")
                        .message(e.getMessage())
                        .errorCode("RESOURCE_NOT_FOUND")
                        .build());
            } catch (Exception e) {
                failed++;
                results.add(BatchDeleteResult.DeleteResultItem.builder()
                        .id(datasetId)
                        .status("failed")
                        .message(e.getMessage())
                        .errorCode("SYSTEM_ERROR")
                        .build());
                log.error("删除数据集失败: datasetId={}", datasetId, e);
            }
        }

        return BatchDeleteResult.builder()
                .total(total)
                .succeeded(succeeded)
                .failed(failed)
                .results(results)
                .build();
    }
}
