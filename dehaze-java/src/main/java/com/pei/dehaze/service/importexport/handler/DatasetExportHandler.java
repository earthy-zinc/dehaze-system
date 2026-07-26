package com.pei.dehaze.service.importexport.handler;

import cn.hutool.core.collection.CollUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysItemFileService;
import com.pei.dehaze.service.importexport.ExportHandler;
import com.pei.dehaze.service.importexport.model.ExportContext;
import com.pei.dehaze.service.importexport.model.ExportDataProvider;
import com.pei.dehaze.service.importexport.model.ExportFieldConfig;
import com.pei.dehaze.service.strategy.ProgressCallback;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.InputStream;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * 数据集导出处理器
 * <p>整合旧 DatasetExportStrategy / ItemDownloadStrategy / BatchDownloadStrategy / CustomExportStrategy 的 ZIP 打包逻辑。
 * <p>通过 queryParams 中的字段区分不同导出场景：
 * <ul>
 *     <li>{@code datasetId}：导出整个数据集的所有数据项</li>
 *     <li>{@code itemId}：导出单个数据项的文件</li>
 *     <li>{@code itemIds}：批量导出多个数据项的文件</li>
 *     <li>{@code filters}：按筛选条件导出数据集内数据项</li>
 * </ul>
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class DatasetExportHandler implements ExportHandler {

    private static final String STRUCTURE_BY_ITEM = "by_item";
    private static final String THUMBNAIL_SUBFOLDER = "thumbnail";
    private static final String DEFAULT_FILE_EXTENSION = ".jpg";
    private static final int ZIP_BUFFER_SIZE = 8192;

    private final SysDatasetItemService datasetItemService;
    private final SysItemFileService itemFileService;
    private final SysFileService fileService;
    private final FileService minioFileService;

    @Override
    public String getModule() {
        return "dataset";
    }

    @Override
    public boolean useDirectExport() {
        return true;
    }

    @Override
    public long estimateCount(Map<String, Object> queryParams) {
        if (queryParams == null || queryParams.isEmpty()) {
            return 0;
        }
        List<SysDatasetItem> items = resolveItems(queryParams);
        if (items.isEmpty()) {
            return 0;
        }
        List<Long> itemIds = items.stream().map(SysDatasetItem::getId).toList();
        return itemFileService.count(new LambdaQueryWrapper<SysItemFile>()
                .in(SysItemFile::getItemId, itemIds));
    }

    @Override
    public void export(ExportContext ctx, ProgressCallback callback) throws Exception {
        Map<String, Object> params = ctx.getQueryParams();
        if (params == null || params.isEmpty()) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "数据集导出参数不能为空");
        }

        ExportOptions options = ExportOptions.from(params);
        List<SysDatasetItem> items = resolveItems(params);
        if (items.isEmpty()) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "未找到可导出的数据项");
        }

        List<Long> itemIds = items.stream().map(SysDatasetItem::getId).toList();
        Map<Long, List<SysItemFile>> itemFilesMap = itemFileService.list(
                        new LambdaQueryWrapper<SysItemFile>()
                                .in(SysItemFile::getItemId, itemIds))
                .stream()
                .collect(Collectors.groupingBy(SysItemFile::getItemId));

        int totalFiles = 0;
        for (SysDatasetItem item : items) {
            int fileCount = itemFilesMap.getOrDefault(item.getId(), Collections.emptyList()).size();
            totalFiles += fileCount;
            if (Boolean.TRUE.equals(options.includeThumbnail)) {
                totalFiles += fileCount;
            }
        }

        callback.updateProgress(0, totalFiles, "开始导出数据集文件");

        int processedFiles = 0;
        try (ZipOutputStream zos = new ZipOutputStream(ctx.getOutputStream())) {
            for (SysDatasetItem item : items) {
                callback.checkCancelled();

                List<SysItemFile> itemFiles = itemFilesMap.getOrDefault(item.getId(), Collections.emptyList());
                for (SysItemFile itemFile : itemFiles) {
                    callback.checkCancelled();

                    if (shouldIncludeType(options.includeTypes, itemFile.getType())) {
                        addFileToZip(zos, itemFile, options.structure, item.getName(), null);
                        processedFiles++;
                        callback.updateProgress(processedFiles, totalFiles, "正在导出: " + item.getName());
                    }

                    if (Boolean.TRUE.equals(options.includeThumbnail)) {
                        addFileToZip(zos, itemFile, options.structure, item.getName(), THUMBNAIL_SUBFOLDER);
                        processedFiles++;
                        callback.updateProgress(processedFiles, totalFiles, "导出缩略图");
                    }
                }
            }
        }

        log.info("数据集导出完成: taskId={}, itemCount={}, fileCount={}",
                ctx.getTaskId(), items.size(), processedFiles);
    }

    @Override
    public List<ExportFieldConfig> getFieldConfigs() {
        return List.of(
                ExportFieldConfig.builder().field("datasetName").label("数据集名称").order(1).build(),
                ExportFieldConfig.builder().field("itemName").label("数据项名称").order(2).build(),
                ExportFieldConfig.builder().field("fileType").label("文件类型").order(3).build(),
                ExportFieldConfig.builder().field("fileName").label("文件名").order(4).build(),
                ExportFieldConfig.builder().field("fileSize").label("文件大小").order(5).build()
        );
    }

    @Override
    public ExportDataProvider getDataProvider(ExportContext ctx) {
        return (pageNum, pageSize) -> List.of();
    }

    /**
     * 根据查询参数解析要导出的数据项列表
     * <p>支持四种场景：整个数据集、单个数据项、批量数据项、按筛选条件
     */
    @SuppressWarnings("unchecked")
    private List<SysDatasetItem> resolveItems(Map<String, Object> params) {
        Object itemIdsObj = params.get("itemIds");
        if (itemIdsObj instanceof List<?> list && !list.isEmpty()) {
            List<Long> itemIds = ((List<Number>) list).stream()
                    .map(Number::longValue)
                    .toList();
            return datasetItemService.listByIds(itemIds);
        }

        Object itemIdObj = params.get("itemId");
        if (itemIdObj != null) {
            Long itemId = ((Number) itemIdObj).longValue();
            SysDatasetItem item = datasetItemService.getById(itemId);
            return item == null ? List.of() : List.of(item);
        }

        Object datasetIdObj = params.get("datasetId");
        if (datasetIdObj == null) {
            datasetIdObj = params.get("targetId");
        }
        if (datasetIdObj != null) {
            Long datasetId = ((Number) datasetIdObj).longValue();
            LambdaQueryWrapper<SysDatasetItem> query = new LambdaQueryWrapper<SysDatasetItem>()
                    .eq(SysDatasetItem::getDatasetId, datasetId);

            Object filtersObj = params.get("filters");
            if (filtersObj instanceof Map<?, ?> filterMap) {
                Object name = filterMap.get("name");
                if (name instanceof String nameStr && !nameStr.isBlank()) {
                    query.like(SysDatasetItem::getName, nameStr);
                }
            }
            return datasetItemService.list(query);
        }

        return List.of();
    }

    private boolean shouldIncludeType(List<String> includeTypes, String type) {
        if (CollUtil.isEmpty(includeTypes)) {
            return true;
        }
        return includeTypes.contains(type);
    }

    private void addFileToZip(ZipOutputStream zos, SysItemFile itemFile,
                              String structure, String itemName, String subfolder) throws Exception {
        SysFile sysFile = fileService.getById(itemFile.getFileId());
        if (sysFile == null || sysFile.getObjectName() == null) {
            log.warn("文件不存在或 objectName 为空: fileId={}", itemFile.getFileId());
            return;
        }

        String entryPath = buildZipEntryPath(structure, itemName, subfolder, itemFile.getId(), sysFile.getName());
        ZipEntry zipEntry = new ZipEntry(entryPath);
        zos.putNextEntry(zipEntry);

        try (InputStream inputStream = minioFileService.downLoadFile(sysFile.getObjectName())) {
            byte[] buffer = new byte[ZIP_BUFFER_SIZE];
            int len;
            while ((len = inputStream.read(buffer)) > 0) {
                zos.write(buffer, 0, len);
            }
        }
        zos.closeEntry();
    }

    private String buildZipEntryPath(String structure, String itemName, String subfolder,
                                     Long fileId, String fileName) {
        String extension = getFileExtension(fileName);
        String baseName = fileId + extension;

        if (STRUCTURE_BY_ITEM.equals(structure)) {
            return subfolder != null
                    ? itemName + "/" + subfolder + "/" + baseName
                    : itemName + "/" + baseName;
        }
        return subfolder != null
                ? subfolder + "/" + baseName
                : baseName;
    }

    private String getFileExtension(String fileName) {
        if (fileName == null) {
            return DEFAULT_FILE_EXTENSION;
        }
        int dotIndex = fileName.lastIndexOf('.');
        return dotIndex > 0 ? fileName.substring(dotIndex) : DEFAULT_FILE_EXTENSION;
    }

    /**
     * 数据集导出选项
     */
    private record ExportOptions(String structure, List<String> includeTypes, Boolean includeThumbnail) {
        static ExportOptions from(Map<String, Object> params) {
            String structure = STRUCTURE_BY_ITEM;
            List<String> includeTypes = null;
            Boolean includeThumbnail = Boolean.FALSE;

            Object optionsObj = params.get("options");
            if (optionsObj instanceof Map<?, ?> optionsMap) {
                if (optionsMap.get("structure") instanceof String s) {
                    structure = s;
                }
                if (optionsMap.get("includeTypes") instanceof List<?> list) {
                    includeTypes = list.stream()
                            .filter(String.class::isInstance)
                            .map(String.class::cast)
                            .toList();
                }
                if (optionsMap.get("includeThumbnail") instanceof Boolean b) {
                    includeThumbnail = b;
                }
            }
            return new ExportOptions(structure, includeTypes, includeThumbnail);
        }
    }
}
