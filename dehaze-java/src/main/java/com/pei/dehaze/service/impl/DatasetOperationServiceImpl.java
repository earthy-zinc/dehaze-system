package com.pei.dehaze.service.impl;

import cn.hutool.core.lang.Assert;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
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
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * 数据集操作服务实现
 * 处理跨服务的复杂组合操作，避免循环依赖
 *
 * @author earthy-zinc
 * @since 2025-12-13
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class DatasetOperationServiceImpl implements DatasetOperationService {

    @Lazy
    private final SysDatasetService sysDatasetService;

    private final SysDatasetItemService sysDatasetItemService;

    private final SysItemFileService sysItemFileService;

    private final FileBOFactory fileBOFactory;

    private final ImageProcessingService imageProcessingService;

    @Override
    public DatasetItemVO createDatasetItemWithImages(DatasetItemUploadForm form) {
        DatasetItemVO result = doCreateDatasetItemWithImages(form);
        sysDatasetService.evictAllDatasetsCache();
        return result;
    }

    private DatasetItemVO doCreateDatasetItemWithImages(DatasetItemUploadForm form) {
        // 清晰图和有雾图均为可选（适配不同数据集规范），但至少上传一张图片
        if (form.getClearImage() == null && (form.getHazyImages() == null || form.getHazyImages().isEmpty())) {
            throw new BusinessException("至少上传一张图片（清晰图或有雾图）");
        }

        // 校验配对图片分辨率一致性（清晰图存在时才校验）
        validatePairedImageResolution(form);

        // 创建数据项
        SysDatasetItem datasetItem = sysDatasetItemService.createDatasetItem(form.getDatasetId(), form.getName());

        // 保存清晰图（可选）
        if (form.getClearImage() != null) {
            ItemFileBO clearItemBO = fileBOFactory.createItemFileBO(
                    form.getClearImage(),
                    "",
                    "clear",
                    "",
                    form.getSceneType(),
                    ""
            );
            sysItemFileService.saveItemFile(datasetItem.getId(), clearItemBO);
        }

        // 保存有雾图（可选，haze_level 支持多种规范，可为空）
        if (form.getHazyImages() != null) {
            for (int i = 0; i < form.getHazyImages().size(); i++) {
                String hazeLevel = form.getHazeLevels() != null && i < form.getHazeLevels().size()
                        ? form.getHazeLevels().get(i) : "";
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
        }

        // 构建返回结果
        return sysDatasetItemService.getDatasetItem(datasetItem.getId());
    }

    /**
     * 校验配对图片分辨率一致性（清晰图存在时才校验）
     */
    private void validatePairedImageResolution(DatasetItemUploadForm form) {
        int[] clearDimensions = null;

        // 校验清晰图（可选）
        if (form.getClearImage() != null) {
            imageProcessingService.validateImageFile(form.getClearImage());
            clearDimensions = imageProcessingService.getImageDimensions(form.getClearImage());
        }

        // 检查每张有雾图的分辨率和格式
        if (form.getHazyImages() != null) {
            for (int i = 0; i < form.getHazyImages().size(); i++) {
                MultipartFile hazyImage = form.getHazyImages().get(i);

                // 校验文件格式和大小
                imageProcessingService.validateImageFile(hazyImage);

                // 校验分辨率（如有清晰图则需一致）
                int[] hazyDimensions = imageProcessingService.getImageDimensions(hazyImage);

                if (clearDimensions != null &&
                        (hazyDimensions[0] != clearDimensions[0] || hazyDimensions[1] != clearDimensions[1])) {
                    throw new BusinessException(ResultCode.PARAM_ERROR, String.format(
                            "配对图片分辨率不一致，清晰图：%dx%d，有雾图%s：%dx%d",
                            clearDimensions[0], clearDimensions[1],
                            hazyImage.getOriginalFilename(),
                            hazyDimensions[0], hazyDimensions[1]
                    ));
                }
            }
        }
    }

    @Override
    public BatchUploadResultVO batchCreateDatasetItemsWithImages(BatchDatasetItemUploadForm form) {
        int successGroups = 0;
        int failedGroups = 0;
        List<BatchUploadSuccessItemVO> successItems = new ArrayList<>();
        List<BatchUploadFailedItemVO> failedItems = new ArrayList<>();

        // 按文件名前缀分组
        Map<String, FileGroup> fileGroups = new HashMap<>();

        for (MultipartFile file : form.getFiles()) {
            String fileName = file.getOriginalFilename();
            if (fileName == null) continue;

            // 提取文件名前缀（去掉下划线和后缀之前的部分）
            String prefix = extractFilePrefix(fileName);

            FileGroup group = fileGroups.computeIfAbsent(prefix, k -> new FileGroup());

            // 判断文件类型
            if (isClearImage(fileName)) {
                group.clearImage = file;
            } else if (isHazyImage(fileName)) {
                // haze_level 支持多种格式，无法解析时为空字符串（不报错）
                String hazeLevel = extractHazeLevel(fileName);
                group.hazyImages.add(new HazyImageInfo(file, hazeLevel));
            } else {
                // 无法识别文件类型的文件归入失败列表
                failedItems.add(new BatchUploadFailedItemVO(fileName,
                        "无法识别文件类型，文件名需包含 clear/gt/clean、hazy/haze 或 trans 关键字"));
            }
        }

        // 处理每个分组
        for (Map.Entry<String, FileGroup> entry : fileGroups.entrySet()) {
            String groupName = entry.getKey();
            FileGroup group = entry.getValue();

            try {
                // 清晰图和有雾图均为可选（适配不同数据集规范）
                if (group.clearImage == null && group.hazyImages.isEmpty()) {
                    throw new BusinessException("分组中未找到任何可识别的图片");
                }

                // 创建配对上传表单
                DatasetItemUploadForm pairForm = new DatasetItemUploadForm();
                pairForm.setDatasetId(form.getDatasetId());
                pairForm.setName(groupName);
                pairForm.setSceneType(form.getSceneType());
                pairForm.setClearImage(group.clearImage);

                List<MultipartFile> hazyImages = new ArrayList<>();
                List<String> hazeLevels = new ArrayList<>();
                for (HazyImageInfo hazyInfo : group.hazyImages) {
                    hazyImages.add(hazyInfo.file());
                    hazeLevels.add(hazyInfo.hazeLevel());
                }

                pairForm.setHazyImages(hazyImages);
                pairForm.setHazeLevels(hazeLevels);

                // 保存配对图片并获取返回结果
                DatasetItemVO createdItem = this.doCreateDatasetItemWithImages(pairForm);
                successGroups++;

                // 记录成功项详情
                int fileCount = (group.clearImage != null ? 1 : 0) + hazyImages.size();
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
        sysDatasetService.evictAllDatasetsCache();
        return result;
    }

    /**
     * 从文件名提取前缀（分组用）。按前导数字分组（如 01_GT.png → "01"），
     * 无前导数字时去除 _clear/_gt/_hazy 等后缀返回剩余部分。
     */
    private String extractFilePrefix(String fileName) {
        String nameWithoutExt = fileName.substring(0, fileName.lastIndexOf('.'));
        // 优先按前导数字分组
        Matcher numMatcher = Pattern.compile("^(\\d+)").matcher(nameWithoutExt);
        if (numMatcher.find()) {
            return numMatcher.group(1);
        }
        // 无前导数字时移除类型后缀
        return nameWithoutExt.replaceAll("(_clear|_gt|_hazy.*|_trans|_depth|_segment)$", "");
    }

    /**
     * 判断是否为清晰图（含 clear/gt/clean 关键字，大小写不敏感）
     */
    private boolean isClearImage(String fileName) {
        String lower = fileName.toLowerCase();
        return lower.contains("clear") || lower.contains("_gt") || lower.contains("clean");
    }

    /**
     * 判断是否为有雾图（含 hazy/haze 关键字，大小写不敏感）
     */
    private boolean isHazyImage(String fileName) {
        String lower = fileName.toLowerCase();
        return lower.contains("hazy") || lower.contains("haze");
    }

    /**
     * 从文件名提取雾霾程度，支持多种规范。无法解析时返回空字符串（表示未标注）。
     *
     * 支持格式：
     * - _hazy_light / _hazy_medium / _hazy_heavy → light/medium/heavy
     * - {id}_{idx}_{beta}.png（如 1000_1_0.74905.png）→ beta=0.74905
     * - {id}_{A}_{beta}.jpg（如 0025_0.8_0.2.jpg）→ beta=0.2（无法可靠区分 A 和 idx，统一取最后一个数值作为 beta）
     * - 无参数后缀（如 01_hazy.png）→ 空字符串
     */
    private String extractHazeLevel(String fileName) {
        // 1. 人工分级：_hazy_light / _hazy_medium / _hazy_heavy
        Matcher levelMatcher = Pattern
                .compile("_hazy_(light|medium|heavy)", Pattern.CASE_INSENSITIVE)
                .matcher(fileName);
        if (levelMatcher.find()) {
            return levelMatcher.group(1).toLowerCase();
        }

        // 2. 学术参数格式：{id}_{idx}_{beta} 或 {id}_{A}_{beta} 等
        //    统一取最后一个数值作为 beta（无法可靠区分 A 和 idx）
        String nameWithoutExt = fileName.substring(0, fileName.lastIndexOf('.'));
        String[] parts = nameWithoutExt.split("_");
        if (parts.length >= 3) {
            List<Double> numParts = new ArrayList<>();
            for (int i = 1; i < parts.length; i++) {
                try {
                    numParts.add(Double.parseDouble(parts[i]));
                } catch (NumberFormatException ignored) {
                }
            }
            if (!numParts.isEmpty()) {
                double beta = numParts.get(numParts.size() - 1);
                return String.format("beta=%s", beta);
            }
        }

        // 3. 无法解析，返回空字符串（表示未标注）
        return "";
    }

    @Override
    public void deleteDatasetItemCascade(Long datasetItemId) {
        Assert.notNull(datasetItemId, "数据项ID不能为空");
        BatchOperationResultVO result = batchDeleteDatasetItemsCascadeWithResult(List.of(datasetItemId));
        if (result.getFailedCount() > 0) {
            BatchActionFailureDetailVO detail = result.getFailureDetails().get(0);
            String reason = detail.getReason();
            if (reason != null && reason.contains("不存在")) {
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, reason);
            }
            throw new BusinessException(reason);
        }
    }

    @Override
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

        // 批量查询所有数据项的文件，避免 N+1 查询
        List<SysItemFile> allFiles = sysItemFileService.list(
                new LambdaQueryWrapper<SysItemFile>()
                        .in(SysItemFile::getItemId, datasetItemIds)
        );
        Map<Long, List<SysItemFile>> filesByItemId = allFiles.stream()
                .collect(Collectors.groupingBy(SysItemFile::getItemId));

        // 批量预检数据项存在性，避免循环内逐条 getById 触发 N+1 查询
        Set<Long> existingItemIds = sysDatasetItemService.listByIds(datasetItemIds).stream()
                .map(SysDatasetItem::getId)
                .collect(Collectors.toSet());

        for (Long datasetItemId : datasetItemIds) {
            try {
                if (!existingItemIds.contains(datasetItemId)) {
                    throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据项不存在");
                }

                List<SysItemFile> itemFiles = filesByItemId.getOrDefault(datasetItemId, Collections.emptyList());
                for (SysItemFile itemFile : itemFiles) {
                    sysItemFileService.deleteFile(itemFile.getId());
                }

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

        // 批量删除成功的数据项，替代循环内逐条 removeById
        if (!successIds.isEmpty()) {
            sysDatasetItemService.removeByIds(successIds);
        }

        sysDatasetService.evictAllDatasetsCache();

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
        List<Long> allDeletedDatasetIds = new ArrayList<>();

        // 批量预检数据集存在性，避免循环内逐条 getById 触发 N+1 查询
        Set<Long> existingDatasetIds = sysDatasetService.listByIds(datasetIds).stream()
                .map(SysDataset::getId)
                .collect(Collectors.toSet());

        for (Long datasetId : datasetIds) {
            try {
                if (!existingDatasetIds.contains(datasetId)) {
                    throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "数据集不存在");
                }

                // 获取该数据集及其所有子数据集的ID
                List<Long> allDatasetIds = sysDatasetService.getDatasetAndDescendantIds(datasetId);
                // 获取所有叶子节点数据集下的数据项
                List<Long> leafDatasetIds = sysDatasetService.getLeafDatasetId(datasetId);

                // 批量查询所有叶子数据集下的数据项，避免 N+1 查询
                List<Long> allItemIds = Collections.emptyList();
                if (!leafDatasetIds.isEmpty()) {
                    List<SysDatasetItem> items = sysDatasetItemService.list(
                            new LambdaQueryWrapper<SysDatasetItem>()
                                    .in(SysDatasetItem::getDatasetId, leafDatasetIds)
                    );
                    allItemIds = items.stream().map(SysDatasetItem::getId).toList();
                }

                // 批量查询所有数据项的文件，避免 N+1 查询
                if (!allItemIds.isEmpty()) {
                    List<SysItemFile> allFiles = sysItemFileService.list(
                            new LambdaQueryWrapper<SysItemFile>()
                                    .in(SysItemFile::getItemId, allItemIds)
                    );
                    // 逐个删除文件（涉及 MinIO 物理文件删除，无法批量）
                    for (SysItemFile itemFile : allFiles) {
                        sysItemFileService.deleteFile(itemFile.getId());
                    }
                }

                // 批量删除数据项
                if (!allItemIds.isEmpty()) {
                    sysDatasetItemService.removeByIds(allItemIds);
                }

                // 收集待删除的数据集 ID（去重），循环结束后统一批量删除
                allDatasetIds = allDatasetIds.stream().distinct().toList();
                allDeletedDatasetIds.addAll(allDatasetIds);

                succeeded++;
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

        // 批量删除所有成功的数据集，替代循环内逐条 removeById
        if (!allDeletedDatasetIds.isEmpty()) {
            sysDatasetService.removeByIds(allDeletedDatasetIds);
        }

        sysDatasetService.evictAllDatasetsCache();

        return BatchDeleteResult.builder()
                .total(total)
                .succeeded(succeeded)
                .failed(failed)
                .results(results)
                .build();
    }

    /** 批量上传时的文件分组，按文件名前缀归类 */
    private static class FileGroup {
        MultipartFile clearImage;
        final List<HazyImageInfo> hazyImages = new ArrayList<>();
    }

    /** 有雾图的文件及其雾霾程度 */
    private record HazyImageInfo(MultipartFile file, String hazeLevel) {}
}
