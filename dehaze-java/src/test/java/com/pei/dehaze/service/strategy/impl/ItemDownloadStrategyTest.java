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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

/**
 * 数据项下载策略单元测试
 * 测试目的：验证 ItemDownloadStrategy 的任务类型、参数校验、执行逻辑
 * 测试范围：
 * 1. getTaskType() 返回正确的任务类型
 * 2. validateParams() 参数校验逻辑
 * 3. execute() 数据项下载执行逻辑（正常、异常场景）
 *
 * @author earthy-zinc
 * @since 2026-01-20
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("数据项下载策略测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class ItemDownloadStrategyTest {

    @InjectMocks
    private ItemDownloadStrategy strategy;

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
        testParams.put("targetId", 200L);
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
        assertEquals(TaskConstants.TYPE_ITEM_DOWNLOAD, taskType);
    }

    // ==================== validateParams 测试 ====================

    @Test
    @DisplayName("validateParams 当 targetId 为 null 时抛出异常")
    void validateParams_WhenTargetIdNull_ThrowsException() {
        // Arrange
        Map<String, Object> params = new HashMap<>();

        // Act & Assert
        BusinessException exception = assertThrows(
            BusinessException.class,
            () -> strategy.validateParams(params)
        );

        assertTrue(exception.getMessage().contains("数据项ID"));
    }

    @Test
    @DisplayName("validateParams 当 targetId 有效时不抛异常")
    void validateParams_WhenTargetIdValid_NoException() {
        // Arrange
        Map<String, Object> params = new HashMap<>();
        params.put("targetId", 200L);

        // Act & Assert
        assertDoesNotThrow(() -> strategy.validateParams(params));
    }

    @Test
    @DisplayName("validateParams 当 targetId 为 0 时有效")
    void validateParams_WhenTargetIdZero_Valid() {
        // Arrange
        Map<String, Object> params = new HashMap<>();
        params.put("targetId", 0);

        // Act & Assert
        assertDoesNotThrow(() -> strategy.validateParams(params));
    }

    // ==================== execute 测试 - 数据项不存在 ====================

    @Test
    @DisplayName("execute 当数据项不存在时返回失败结果")
    void execute_WhenItemNotExists_ReturnsFailure() {
        // Arrange
        when(sysDatasetItemService.getById(200L)).thenReturn(null);

        // Act
        TaskResult result = strategy.execute(testTask, testParams, progressCallback);

        // Assert
        assertFalse(result.isSuccess());
        assertTrue(result.getErrorMessage().contains("数据项不存在"));
        assertNull(result.getData());

        verify(sysDatasetItemService).getById(200L);
        verify(sysItemFileService, never()).list(any(LambdaQueryWrapper.class));
    }

    // ==================== execute 测试 - 数据项无文件 ====================

    @Test
    @DisplayName("execute 当数据项无文件时返回失败结果")
    void execute_WhenItemNoFiles_ReturnsFailure() {
        // Arrange
        SysDatasetItem item = new SysDatasetItem();
        item.setId(200L);
        item.setName("数据项200");

        when(sysDatasetItemService.getById(200L)).thenReturn(item);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(List.of());

        // Act
        TaskResult result = strategy.execute(testTask, testParams, progressCallback);

        // Assert
        assertFalse(result.isSuccess());
        assertTrue(result.getErrorMessage().contains("数据项无文件"));
        verify(sysItemFileService).list(any(LambdaQueryWrapper.class));
        verify(fileService, never()).uploadFile(any(), any(), anyLong(), any());
    }

    // ==================== execute 测试 - 成功执行 ====================

    @Test
    @DisplayName("execute 成功执行时返回正确的结果和元数据")
    void execute_Success_ReturnsCorrectResultWithMetadata() throws Exception {
        // Arrange
        SysDatasetItem item = new SysDatasetItem();
        item.setId(200L);
        item.setName("数据项200");

        SysItemFile itemFile = new SysItemFile();
        itemFile.setId(10L);
        itemFile.setItemId(200L);
        itemFile.setFileId(100L);
        itemFile.setType("image");

        com.pei.dehaze.model.entity.SysFile sysFile = new com.pei.dehaze.model.entity.SysFile();
        sysFile.setId(100L);
        sysFile.setObjectName("test/image.jpg");
        sysFile.setName("image.jpg");

        when(sysDatasetItemService.getById(200L)).thenReturn(item);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(List.of(itemFile));
        when(sysFileServiceDep.getById(100L)).thenReturn(sysFile);
        when(fileService.downLoadFile("test/image.jpg")).thenReturn(new ByteArrayInputStream(new byte[1024]));
        when(fileService.uploadFile(any(), any(), anyLong(), any())).thenReturn("http://example.com/download.zip");

        // Act
        TaskResult result = strategy.execute(testTask, testParams, progressCallback);

        // Assert
        assertTrue(result.isSuccess());
        assertNotNull(result.getData());
        assertTrue(result.getData().toString().contains("http://example.com/download.zip"));
        assertNotNull(result.getMetadata());

        @SuppressWarnings("unchecked")
        Map<String, Object> metadata = result.getMetadata();
        assertEquals(200L, metadata.get("itemId"));
        assertEquals("数据项200", metadata.get("itemName"));
        assertEquals(1, metadata.get("fileCount"));

        verify(progressCallback).updateProgress(eq(0), eq(1), anyString());
        verify(fileService).uploadFile(any(), any(), anyLong(), eq("application/zip"));
    }

    @Test
    @DisplayName("execute 当文件上传失败时返回失败结果")
    void execute_WhenFileUploadFails_ReturnsFailure() throws Exception {
        // Arrange
        SysDatasetItem item = new SysDatasetItem();
        item.setId(200L);
        item.setName("数据项200");

        SysItemFile itemFile = new SysItemFile();
        itemFile.setId(10L);
        itemFile.setItemId(200L);
        itemFile.setFileId(100L);
        itemFile.setType("image");

        com.pei.dehaze.model.entity.SysFile sysFile = new com.pei.dehaze.model.entity.SysFile();
        sysFile.setObjectName("test/image.jpg");
        sysFile.setName("image.jpg");

        when(sysDatasetItemService.getById(200L)).thenReturn(item);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(List.of(itemFile));
        when(sysFileServiceDep.getById(100L)).thenReturn(sysFile);
        when(fileService.downLoadFile(eq("test/image.jpg"))).thenReturn(new ByteArrayInputStream(new byte[1024]));
        when(fileService.uploadFile(any(), any(), anyLong(), any()))
            .thenThrow(new RuntimeException("上传失败"));

        // Act
        TaskResult result = strategy.execute(testTask, testParams, progressCallback);

        // Assert
        assertFalse(result.isSuccess());
        assertTrue(result.getErrorMessage().contains("下载失败"));
        assertTrue(result.getErrorMessage().contains("上传失败"));
    }

    // ==================== 边界场景测试 ====================

    @Test
    @DisplayName("execute 当 options 为 null 时使用默认配置")
    void execute_WhenOptionsNull_UsesDefaultConfig() throws Exception {
        // Arrange
        Map<String, Object> paramsWithoutOptions = new HashMap<>();
        paramsWithoutOptions.put("targetId", 200L);

        SysDatasetItem item = new SysDatasetItem();
        item.setId(200L);
        item.setName("数据项200");

        SysItemFile itemFile = new SysItemFile();
        itemFile.setId(10L);
        itemFile.setItemId(200L);
        itemFile.setFileId(100L);
        itemFile.setType("image");

        com.pei.dehaze.model.entity.SysFile sysFile = new com.pei.dehaze.model.entity.SysFile();
        sysFile.setObjectName("test/image.jpg");
        sysFile.setName("image.jpg");

        when(sysDatasetItemService.getById(200L)).thenReturn(item);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(List.of(itemFile));
        when(sysFileServiceDep.getById(100L)).thenReturn(sysFile);
        when(fileService.downLoadFile(eq("test/image.jpg"))).thenReturn(new ByteArrayInputStream(new byte[1024]));
        when(fileService.uploadFile(any(), any(), anyLong(), any())).thenReturn("http://example.com/download.zip");

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
        paramsWithThumbnail.put("targetId", 200L);
        paramsWithThumbnail.put("options", Map.of(
            "includeThumbnail", true
        ));

        SysDatasetItem item = new SysDatasetItem();
        item.setId(200L);
        item.setName("数据项200");

        SysItemFile itemFile = new SysItemFile();
        itemFile.setId(10L);
        itemFile.setItemId(200L);
        itemFile.setFileId(100L);
        itemFile.setType("image");

        com.pei.dehaze.model.entity.SysFile sysFile = new com.pei.dehaze.model.entity.SysFile();
        sysFile.setObjectName("test/image.jpg");
        sysFile.setName("image.jpg");

        when(sysDatasetItemService.getById(200L)).thenReturn(item);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(List.of(itemFile));
        when(sysFileServiceDep.getById(100L)).thenReturn(sysFile);
        when(fileService.downLoadFile(eq("test/image.jpg"))).thenReturn(new ByteArrayInputStream(new byte[1024]));
        when(fileService.uploadFile(any(), any(), anyLong(), any())).thenReturn("http://example.com/download.zip");

        // Act
        TaskResult result = strategy.execute(testTask, paramsWithThumbnail, progressCallback);

        // Assert
        assertTrue(result.isSuccess());
        verify(progressCallback).updateProgress(eq(0), eq(2), anyString());
    }

    @Test
    @DisplayName("execute 当过滤文件类型时只导出匹配的文件")
    void execute_WhenFilteringFileTypes_ExportsMatchingFilesOnly() throws Exception {
        // Arrange
        Map<String, Object> paramsWithTypes = new HashMap<>();
        paramsWithTypes.put("targetId", 200L);
        paramsWithTypes.put("options", Map.of(
            "includeTypes", List.of("image")
        ));

        SysDatasetItem item = new SysDatasetItem();
        item.setId(200L);
        item.setName("数据项200");

        SysItemFile imageFile = new SysItemFile();
        imageFile.setId(10L);
        imageFile.setItemId(200L);
        imageFile.setFileId(100L);
        imageFile.setType("image");

        SysItemFile videoFile = new SysItemFile();
        videoFile.setId(11L);
        videoFile.setItemId(200L);
        videoFile.setFileId(101L);
        videoFile.setType("video");

        com.pei.dehaze.model.entity.SysFile sysFile = new com.pei.dehaze.model.entity.SysFile();
        sysFile.setObjectName("test/image.jpg");
        sysFile.setName("image.jpg");

        when(sysDatasetItemService.getById(200L)).thenReturn(item);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(List.of(imageFile, videoFile));
        when(sysFileServiceDep.getById(100L)).thenReturn(sysFile);
        when(fileService.downLoadFile(eq("test/image.jpg"))).thenReturn(new ByteArrayInputStream(new byte[1024]));
        when(fileService.uploadFile(any(), any(), anyLong(), any())).thenReturn("http://example.com/download.zip");

        // Act
        TaskResult result = strategy.execute(testTask, paramsWithTypes, progressCallback);

        // Assert
        assertTrue(result.isSuccess());
        // 只有 image 类型的文件被下载（video类型被过滤）
        verify(fileService).downLoadFile(eq("test/image.jpg"));
    }

    @Test
    @DisplayName("execute 当不指定文件类型时导出所有文件")
    void execute_WhenNoFileTypeFilter_ExportsAllFiles() throws Exception {
        // Arrange
        Map<String, Object> paramsWithoutTypes = new HashMap<>();
        paramsWithoutTypes.put("targetId", 200L);
        paramsWithoutTypes.put("options", Map.of());

        SysDatasetItem item = new SysDatasetItem();
        item.setId(200L);
        item.setName("数据项200");

        SysItemFile itemFile = new SysItemFile();
        itemFile.setId(10L);
        itemFile.setItemId(200L);
        itemFile.setFileId(100L);
        itemFile.setType("video");

        com.pei.dehaze.model.entity.SysFile sysFile = new com.pei.dehaze.model.entity.SysFile();
        sysFile.setObjectName("test/image.jpg");
        sysFile.setName("image.jpg");

        when(sysDatasetItemService.getById(200L)).thenReturn(item);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(List.of(itemFile));
        when(sysFileServiceDep.getById(100L)).thenReturn(sysFile);
        when(fileService.downLoadFile(eq("test/image.jpg"))).thenReturn(new ByteArrayInputStream(new byte[1024]));
        when(fileService.uploadFile(any(), any(), anyLong(), any())).thenReturn("http://example.com/download.zip");

        // Act
        TaskResult result = strategy.execute(testTask, paramsWithoutTypes, progressCallback);

        // Assert
        assertTrue(result.isSuccess());
        verify(fileService).uploadFile(any(), any(), anyLong(), eq("application/zip"));
    }
}
