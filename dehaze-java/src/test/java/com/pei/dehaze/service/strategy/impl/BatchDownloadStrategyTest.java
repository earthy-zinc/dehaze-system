package com.pei.dehaze.service.strategy.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysItemFileService;
import com.pei.dehaze.service.strategy.ProgressCallback;
import com.pei.dehaze.service.strategy.TaskResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

/**
 * 批量下载策略单元测试
 * 测试目的：验证 BatchDownloadStrategy 的任务类型、参数校验、执行逻辑
 * 测试范围：
 * 1. getTaskType() 返回正确的任务类型
 * 2. validateParams() 参数校验逻辑（null、空列表、有效列表）
 * 3. execute() 批量下载执行逻辑（正常、异常场景）
 *
 * @author earthy-zinc
 * @since 2026-01-20
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("批量下载策略测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class BatchDownloadStrategyTest {

    @InjectMocks
    private BatchDownloadStrategy strategy;

    @Mock
    private SysDatasetItemService sysDatasetItemService;

    @Mock
    private SysItemFileService sysItemFileService;

    @Mock
    private FileService fileService;

    @Mock
    private com.pei.dehaze.service.SysFileService sysFileServiceDep;

    @Mock
    private ProgressCallback progressCallback;

    private SysTask testTask;
    private Map<String, Object> testParams;

    @BeforeEach
    void setUp() {
        // 注入依赖
        ReflectionTestUtils.setField(strategy, "fileService", fileService);
        ReflectionTestUtils.setField(strategy, "sysFileService", sysFileServiceDep);

        testTask = new SysTask();
        testTask.setTaskId("1");

        testParams = new HashMap<>();
        testParams.put("targetIds", List.of(300L, 301L, 302L));
        testParams.put("options", Map.of(
            "structure", "by_item",
            "includeTypes", List.of("image"),
            "includeThumbnail", false
        ));
    }

    // ==================== getTaskType 测试 ====================

    @Test
    @DisplayName("getTaskType 返回正确的任务类型")
    void getTaskType_ReturnsCorrectType() {
        // Act
        String taskType = strategy.getTaskType();

        // Assert
        assertEquals(TaskConstants.TYPE_BATCH_DOWNLOAD, taskType);
    }

    // ==================== validateParams 测试 ====================

    @Test
    @DisplayName("validateParams 当 targetIds 为 null 时抛出异常")
    void validateParams_WhenTargetIdsNull_ThrowsException() {
        // Arrange
        Map<String, Object> params = new HashMap<>();

        // Act & Assert
        BusinessException exception = assertThrows(
            BusinessException.class,
            () -> strategy.validateParams(params)
        );

        assertTrue(exception.getMessage().contains("数据项ID列表"));
    }

    @Test
    @DisplayName("validateParams 当 targetIds 为空列表时抛出异常")
    void validateParams_WhenTargetIdsEmpty_ThrowsException() {
        // Arrange
        Map<String, Object> params = new HashMap<>();
        params.put("targetIds", List.of());

        // Act & Assert
        BusinessException exception = assertThrows(
            BusinessException.class,
            () -> strategy.validateParams(params)
        );

        assertTrue(exception.getMessage().contains("数据项ID列表"));
    }

    @Test
    @DisplayName("validateParams 当 targetIds 有效时不抛异常")
    void validateParams_WhenTargetIdsValid_NoException() {
        // Arrange
        Map<String, Object> params = new HashMap<>();
        params.put("targetIds", List.of(300L, 301L));

        // Act & Assert
        assertDoesNotThrow(() -> strategy.validateParams(params));
    }

    @Test
    @DisplayName("validateParams 当 targetIds 为单个ID时有效")
    void validateParams_WhenTargetIdsSingleItem_Valid() {
        // Arrange
        Map<String, Object> params = new HashMap<>();
        params.put("targetIds", List.of(300L));

        // Act & Assert
        assertDoesNotThrow(() -> strategy.validateParams(params));
    }

    // ==================== execute 测试 - 所有数据项不存在 ====================

    @Test
    @DisplayName("execute 当所有数据项不存在时返回失败结果")
    void execute_WhenAllItemsNotExists_ReturnsFailure() {
        // Arrange
        when(sysDatasetItemService.listByIds(any())).thenReturn(List.of());

        // Act
        TaskResult result = strategy.execute(testTask, testParams, progressCallback);

        // Assert
        assertFalse(result.isSuccess());
        assertTrue(result.getErrorMessage().contains("未找到有效的数据项"));
        assertNull(result.getData());

        verify(sysDatasetItemService).listByIds(any());
        verify(sysItemFileService, never()).list(any(LambdaQueryWrapper.class));
        verify(fileService, never()).uploadFile(any(), any(), anyLong(), any());
    }

    // ==================== execute 测试 - 成功执行 ====================

    @Test
    @DisplayName("execute 成功执行时返回正确的结果和元数据")
    void execute_Success_ReturnsCorrectResultWithMetadata() throws Exception {
        // Arrange
        List<SysDatasetItem> items = new ArrayList<>();
        for (Long id : List.of(300L, 301L, 302L)) {
            SysDatasetItem item = new SysDatasetItem();
            item.setId(id);
            item.setName("数据项" + id);
            items.add(item);
        }

        List<SysItemFile> itemFiles = new ArrayList<>();
        for (Long id : List.of(300L, 301L, 302L)) {
            SysItemFile itemFile = new SysItemFile();
            itemFile.setId(id);
            itemFile.setItemId(id);
            itemFile.setFileId(Long.valueOf(id * 100));
            itemFile.setType("image");
            itemFiles.add(itemFile);
        }

        when(sysDatasetItemService.listByIds(any())).thenReturn(items);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(itemFiles);
        when(sysItemFileService.count(any(LambdaQueryWrapper.class))).thenReturn(1L);
        when(sysFileServiceDep.getById(any())).thenReturn(createMockSysFile());
        when(fileService.downLoadFile(any())).thenReturn(new ByteArrayInputStream(new byte[1024]));
        when(fileService.uploadFile(any(), any(), anyLong(), any())).thenReturn("http://example.com/batch.zip");

        // Act
        TaskResult result = strategy.execute(testTask, testParams, progressCallback);

        // Assert
        assertTrue(result.isSuccess());
        assertNotNull(result.getData());
        assertTrue(result.getData().toString().contains("http://example.com/batch.zip"));
        assertNotNull(result.getMetadata());

        @SuppressWarnings("unchecked")
        Map<String, Object> metadata = result.getMetadata();
        assertEquals(3, metadata.get("itemCount"));
        assertEquals(3, metadata.get("requestedCount"));

        verify(progressCallback).updateProgress(eq(0), eq(3), anyString());
        verify(fileService).uploadFile(any(), any(), anyLong(), eq("application/zip"));
    }

    @Test
    @DisplayName("execute 当部分数据项不存在时只导出存在的数据项")
    void execute_WhenSomeItemsNotFound_ExistsOnlyExistingItems() throws Exception {
        // Arrange
        List<SysDatasetItem> items = new ArrayList<>();
        SysDatasetItem item1 = new SysDatasetItem();
        item1.setId(300L);
        item1.setName("数据项300");
        items.add(item1);

        List<SysItemFile> itemFiles = new ArrayList<>();
        SysItemFile itemFile = new SysItemFile();
        itemFile.setId(10L);
        itemFile.setItemId(300L);
        itemFile.setFileId(100L);
        itemFile.setType("image");
        itemFiles.add(itemFile);

        when(sysDatasetItemService.listByIds(any())).thenReturn(items);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(itemFiles);
        when(sysItemFileService.count(any(LambdaQueryWrapper.class))).thenReturn(1L);
        when(sysFileServiceDep.getById(any())).thenReturn(createMockSysFile());
        when(fileService.downLoadFile(any())).thenReturn(new ByteArrayInputStream(new byte[1024]));
        when(fileService.uploadFile(any(), any(), anyLong(), any())).thenReturn("http://example.com/batch.zip");

        // Act
        TaskResult result = strategy.execute(testTask, testParams, progressCallback);

        // Assert
        assertTrue(result.isSuccess());
        @SuppressWarnings("unchecked")
        Map<String, Object> metadata = result.getMetadata();
        assertEquals(1, metadata.get("itemCount"));
        assertEquals(3, metadata.get("requestedCount"));
    }

    @Test
    @DisplayName("execute 当文件上传失败时返回失败结果")
    void execute_WhenFileUploadFails_ReturnsFailure() throws Exception {
        // Arrange
        SysDatasetItem item = new SysDatasetItem();
        item.setId(300L);
        item.setName("数据项300");

        when(sysDatasetItemService.listByIds(any())).thenReturn(List.of(item));
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(List.of());
        when(sysItemFileService.count(any(LambdaQueryWrapper.class))).thenReturn(0L);

        when(fileService.uploadFile(any(), any(), anyLong(), any()))
            .thenThrow(new RuntimeException("上传失败"));

        // Act
        TaskResult result = strategy.execute(testTask, testParams, progressCallback);

        // Assert
        assertFalse(result.isSuccess());
        assertTrue(result.getErrorMessage().contains("批量下载失败"));
        assertTrue(result.getErrorMessage().contains("上传失败"));
    }

    // ==================== 边界场景测试 ====================

    @Test
    @DisplayName("execute 当 options 为 null 时使用默认配置")
    void execute_WhenOptionsNull_UsesDefaultConfig() throws Exception {
        // Arrange
        Map<String, Object> paramsWithoutOptions = new HashMap<>();
        paramsWithoutOptions.put("targetIds", List.of(300L));

        SysDatasetItem item = new SysDatasetItem();
        item.setId(300L);
        item.setName("数据项300");

        when(sysDatasetItemService.listByIds(any())).thenReturn(List.of(item));
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(List.of());
        when(sysItemFileService.count(any(LambdaQueryWrapper.class))).thenReturn(0L);
        when(fileService.uploadFile(any(), any(), anyLong(), any())).thenReturn("http://example.com/batch.zip");

        // Act
        TaskResult result = strategy.execute(testTask, paramsWithoutOptions, progressCallback);

        // Assert
        assertTrue(result.isSuccess());
        assertNotNull(result.getData());
    }

    @Test
    @DisplayName("execute 当包含缩略图时正确计算文件数")
    void execute_WhenIncludeThumbnail_CalculatesCorrectFileCount() throws Exception {
        // Arrange
        Map<String, Object> paramsWithThumbnail = new HashMap<>();
        paramsWithThumbnail.put("targetIds", List.of(300L));
        paramsWithThumbnail.put("options", Map.of(
            "includeThumbnail", true
        ));

        SysDatasetItem item = new SysDatasetItem();
        item.setId(300L);
        item.setName("数据项300");

        SysItemFile itemFile = new SysItemFile();
        itemFile.setId(10L);
        itemFile.setItemId(300L);
        itemFile.setFileId(100L);
        itemFile.setType("image");

        when(sysDatasetItemService.listByIds(any())).thenReturn(List.of(item));
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(List.of(itemFile));
        when(sysItemFileService.count(any(LambdaQueryWrapper.class))).thenReturn(1L);
        when(sysFileServiceDep.getById(any())).thenReturn(createMockSysFile());
        when(fileService.downLoadFile(any())).thenReturn(new ByteArrayInputStream(new byte[1024]));
        when(fileService.uploadFile(any(), any(), anyLong(), any())).thenReturn("http://example.com/batch.zip");

        // Act
        TaskResult result = strategy.execute(testTask, paramsWithThumbnail, progressCallback);

        // Assert
        assertTrue(result.isSuccess());
        verify(progressCallback).updateProgress(eq(0), eq(2), anyString());
    }

    @Test
    @DisplayName("execute 当批量下载大量数据项时正确处理")
    void execute_WhenDownloadingManyItems_HandlesCorrectly() throws Exception {
        // Arrange
        List<Long> itemIds = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            itemIds.add((long) (400 + i));
        }
        testParams.put("targetIds", itemIds);

        List<SysDatasetItem> items = new ArrayList<>();
        for (Long id : itemIds) {
            SysDatasetItem item = new SysDatasetItem();
            item.setId(id);
            item.setName("数据项" + id);
            items.add(item);
        }

        when(sysDatasetItemService.listByIds(any())).thenReturn(items);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(List.of());
        when(sysItemFileService.count(any(LambdaQueryWrapper.class))).thenReturn(1L);
        when(fileService.uploadFile(any(), any(), anyLong(), any())).thenReturn("http://example.com/batch.zip");

        // Act
        TaskResult result = strategy.execute(testTask, testParams, progressCallback);

        // Assert
        assertTrue(result.isSuccess());
        @SuppressWarnings("unchecked")
        Map<String, Object> metadata = result.getMetadata();
        assertEquals(10, metadata.get("itemCount"));
        assertEquals(10, metadata.get("requestedCount"));
    }

    @Test
    @DisplayName("execute 当数据项有不同数量的文件时正确计算总数")
    void execute_WhenItemsHaveDifferentFileCounts_CalculatesCorrectTotal() throws Exception {
        // Arrange
        List<Long> itemIds = List.of(300L, 301L);
        testParams.put("targetIds", itemIds);

        List<SysDatasetItem> items = new ArrayList<>();
        for (Long id : itemIds) {
            SysDatasetItem item = new SysDatasetItem();
            item.setId(id);
            item.setName("数据项" + id);
            items.add(item);
        }

        // 模拟不同数据项有不同数量的文件: 300 有 1 个文件, 301 有 3 个文件, 总共 4 个
        List<SysItemFile> itemFiles = new ArrayList<>();
        SysItemFile f1 = new SysItemFile();
        f1.setId(10L); f1.setItemId(300L); f1.setFileId(100L); f1.setType("image");
        itemFiles.add(f1);
        for (long i = 1; i <= 3; i++) {
            SysItemFile f = new SysItemFile();
            f.setId(20L + i); f.setItemId(301L); f.setFileId(200L + i); f.setType("image");
            itemFiles.add(f);
        }

        when(sysDatasetItemService.listByIds(any())).thenReturn(items);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(itemFiles);
        when(sysFileServiceDep.getById(any())).thenReturn(createMockSysFile());
        when(fileService.downLoadFile(any())).thenReturn(new ByteArrayInputStream(new byte[1024]));
        when(fileService.uploadFile(any(), any(), anyLong(), any())).thenReturn("http://example.com/batch.zip");

        // Act
        TaskResult result = strategy.execute(testTask, testParams, progressCallback);

        // Assert
        assertTrue(result.isSuccess());
        // 2个数据项，共4个文件
        verify(progressCallback).updateProgress(eq(0), eq(4), anyString());
    }

    /**
     * 创建模拟的 SysFile 对象
     */
    private com.pei.dehaze.model.entity.SysFile createMockSysFile() {
        com.pei.dehaze.model.entity.SysFile sysFile = new com.pei.dehaze.model.entity.SysFile();
        sysFile.setObjectName("test/image.jpg");
        sysFile.setName("image.jpg");
        return sysFile;
    }
}
