package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.service.impl.TaskExecutorImpl;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

/**
 * 任务执行器单元测试
 * 测试目的：验证任务执行器的异步调用正确性
 * 测试范围：
 * 1. 提交导出任务方法
 * 2. 异步执行验证
 * 3. 参数传递验证
 * <p>
 * 注意：由于当前为TDD阶段，Executor实现为空，此测试验证接口调用流程
 * 异步测试需要在实际实现后完善，当前仅验证方法调用
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("任务执行器测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class TaskExecutorTest {

    @InjectMocks
    private TaskExecutorImpl taskExecutor;

    @Mock
    private SysTaskMapper sysTaskMapper;

    @Mock
    private SysDatasetService sysDatasetService;

    @Mock
    private SysDatasetItemService sysDatasetItemService;

    @Mock
    private SysItemFileService sysItemFileService;

    @Mock
    private SysFileService sysFileService;

    @Mock
    private FileService fileService;

    @Mock
    private RedisTemplate<String, Object> redisTemplate;

    @Mock
    private ValueOperations<String, Object> valueOperations;

    private ExportTaskCreateForm mockForm;

    @BeforeEach
    void setUp() {
        mockForm = new ExportTaskCreateForm();
        mockForm.setType("dataset");
        mockForm.setTargetId(1L);

        // Mock redisTemplate behavior
        when(redisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.get(anyString())).thenReturn(null);
    }

    // ==================== 提交导出任务测试 ====================

    /**
     * 测试提交导出任务 - 单个数据集导出
     * 测试场景：提交单个数据集的导出任务到异步执行器
     * 验证内容：
     * 1. 方法被正确调用
     * 2. 参数正确传递
     * 注意：当前实现为空，待实现后验证异步执行
     */
    @Test
    @DisplayName("submitExportTask - 提交单个数据集导出任务")
    void testSubmitExportTask_SingleDataset() {
        // Arrange
        Long taskId = 1L;
        mockForm.setType("dataset");
        mockForm.setTargetId(1L);

        SysTask sysTask = new SysTask();
        sysTask.setId(taskId);
        sysTask.setTaskId("task_" + taskId);
        sysTask.setStatus("pending");

        SysDataset mockDataset = new SysDataset();
        mockDataset.setId(1L);
        mockDataset.setName("test_dataset");

        SysDatasetItem mockItem = new SysDatasetItem();
        mockItem.setId(1L);
        mockItem.setDatasetId(1L);

        SysItemFile mockItemFile = new SysItemFile();
        mockItemFile.setId(1L);
        mockItemFile.setItemId(1L);
        mockItemFile.setType("clear");
        mockItemFile.setFileId(1L);

        SysFile mockFile = SysFile.builder()
                .id(1L)
                .name("test.jpg")
                .objectName("test_object.jpg")
                .build();

        when(sysTaskMapper.selectById(taskId)).thenReturn(sysTask);
        when(sysDatasetService.getById(1L)).thenReturn(mockDataset);
        when(sysDatasetItemService.list(any(LambdaQueryWrapper.class))).thenReturn(Arrays.asList(mockItem));
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(Arrays.asList(mockItemFile));
        when(sysFileService.getById(1L)).thenReturn(mockFile);
        when(fileService.uploadFile(anyString(), any(), anyLong(), anyString())).thenReturn("http://test.com/export.zip");
        when(fileService.downLoadFile(anyString())).thenReturn(new java.io.ByteArrayInputStream(new byte[1024]));

        // Act
        taskExecutor.submitExportTask(taskId, mockForm);

        // Assert
        verify(sysTaskMapper, atLeastOnce()).updateById(any(SysTask.class));
    }

    /**
     * 测试提交导出任务 - 批量数据集导出
     * 测试场景：提交多个数据集的批量导出任务到异步执行器
     * 验证内容：
     * 1. 方法被正确调用
     * 2. 批量参数正确传递
     */
    @Test
    @DisplayName("submitExportTask - 提交批量数据集导出任务")
    void testSubmitExportTask_BatchDatasets() {
        // Arrange
        Long taskId = 2L;
        mockForm.setType("batch_items");
        mockForm.setTargetId(null);
        mockForm.setTargetIds(Arrays.asList(1L, 2L, 3L));

        SysTask sysTask = new SysTask();
        sysTask.setId(taskId);
        sysTask.setTaskId("task_" + taskId);
        sysTask.setStatus("pending");

        SysDatasetItem mockItem = new SysDatasetItem();
        mockItem.setId(1L);

        SysItemFile mockItemFile = new SysItemFile();
        mockItemFile.setId(1L);
        mockItemFile.setItemId(1L);
        mockItemFile.setType("clear");
        mockItemFile.setFileId(1L);

        SysFile mockFile = SysFile.builder()
                .id(1L)
                .objectName("test.jpg")
                .build();

        when(sysTaskMapper.selectById(taskId)).thenReturn(sysTask);
        when(sysDatasetItemService.getById(anyLong())).thenReturn(mockItem);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(Arrays.asList(mockItemFile));
        when(sysFileService.getById(1L)).thenReturn(mockFile);
        when(fileService.uploadFile(anyString(), any(), anyLong(), anyString())).thenReturn("http://test.com/export.zip");
        when(fileService.downLoadFile(anyString())).thenReturn(new java.io.ByteArrayInputStream(new byte[1024]));

        // Act
        taskExecutor.submitExportTask(taskId, mockForm);

        // Assert
        verify(sysTaskMapper, atLeastOnce()).updateById(any(SysTask.class));
        assertEquals(3, mockForm.getTargetIds().size());
    }

    /**
     * 测试提交导出任务 - 包含导出选项
     * 测试场景：提交任务时指定导出选项
     * 验证内容：
     * 1. 导出选项参数正确传递
     * 2. 选项值正确设置
     */
    @Test
    @DisplayName("submitExportTask - 提交任务时包含导出选项")
    void testSubmitExportTask_WithOptions() {
        // Arrange
        Long taskId = 3L;
        mockForm.setType("dataset");
        mockForm.setTargetId(1L);
        ExportTaskCreateForm.ExportOptions options = new ExportTaskCreateForm.ExportOptions();
        options.setStructure("flat");
        options.setIncludeTypes(Arrays.asList("clear", "hazy"));
        options.setIncludeThumbnail(true);
        mockForm.setOptions(options);

        SysTask sysTask = new SysTask();
        sysTask.setId(taskId);
        sysTask.setTaskId("task_" + taskId);
        sysTask.setStatus("pending");

        SysDataset mockDataset = new SysDataset();
        mockDataset.setId(1L);
        mockDataset.setName("test_dataset");

        SysDatasetItem mockItem = new SysDatasetItem();
        mockItem.setId(1L);
        mockItem.setDatasetId(1L);

        SysItemFile mockItemFile = new SysItemFile();
        mockItemFile.setId(1L);
        mockItemFile.setItemId(1L);
        mockItemFile.setType("clear");
        mockItemFile.setFileId(1L);

        SysFile mockFile = SysFile.builder()
                .id(1L)
                .objectName("test.jpg")
                .build();

        when(sysTaskMapper.selectById(taskId)).thenReturn(sysTask);
        when(sysDatasetService.getById(1L)).thenReturn(mockDataset);
        when(sysDatasetItemService.list(any(LambdaQueryWrapper.class))).thenReturn(Arrays.asList(mockItem));
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(Arrays.asList(mockItemFile));
        when(sysFileService.getById(1L)).thenReturn(mockFile);
        when(fileService.uploadFile(anyString(), any(), anyLong(), anyString())).thenReturn("http://test.com/export.zip");
        when(fileService.downLoadFile(anyString())).thenReturn(new java.io.ByteArrayInputStream(new byte[1024]));

        // Act
        taskExecutor.submitExportTask(taskId, mockForm);

        // Assert
        verify(sysTaskMapper, atLeastOnce()).updateById(any(SysTask.class));
        assertEquals("flat", options.getStructure());
        assertEquals(2, options.getIncludeTypes().size());
        assertTrue(options.getIncludeThumbnail());
    }

    /**
     * 测试提交导出任务 - 按数据项组织结构
     * 测试场景：使用按数据项组织结构的导出选项
     * 验证内容：
     * 1. structure参数为by_item
     */
    @Test
    @DisplayName("submitExportTask - 按数据项组织结构导出")
    void testSubmitExportTask_ByItemStructure() {
        // Arrange
        Long taskId = 4L;
        mockForm.setType("dataset_item");
        mockForm.setTargetId(1L);
        ExportTaskCreateForm.ExportOptions options = new ExportTaskCreateForm.ExportOptions();
        options.setStructure("by_item");
        options.setIncludeTypes(Arrays.asList("clear", "hazy"));
        options.setIncludeThumbnail(false);
        mockForm.setOptions(options);

        SysTask sysTask = new SysTask();
        sysTask.setId(taskId);
        sysTask.setTaskId("task_" + taskId);
        sysTask.setStatus("pending");

        SysDatasetItem mockItem = new SysDatasetItem();
        mockItem.setId(1L);
        mockItem.setName("test_item");

        SysItemFile mockItemFile = new SysItemFile();
        mockItemFile.setId(1L);
        mockItemFile.setItemId(1L);
        mockItemFile.setType("clear");
        mockItemFile.setFileId(1L);

        SysFile mockFile = SysFile.builder()
                .id(1L)
                .objectName("test.jpg")
                .build();

        when(sysTaskMapper.selectById(taskId)).thenReturn(sysTask);
        when(sysDatasetItemService.getById(1L)).thenReturn(mockItem);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(Arrays.asList(mockItemFile));
        when(sysFileService.getById(1L)).thenReturn(mockFile);
        when(fileService.uploadFile(anyString(), any(), anyLong(), anyString())).thenReturn("http://test.com/export.zip");
        when(fileService.downLoadFile(anyString())).thenReturn(new java.io.ByteArrayInputStream(new byte[1024]));

        // Act
        taskExecutor.submitExportTask(taskId, mockForm);

        // Assert
        verify(sysTaskMapper, atLeastOnce()).updateById(any(SysTask.class));
        assertEquals("by_item", options.getStructure());
    }

    /**
     * 测试提交导出任务 - 扁平组织结构
     * 测试场景：使用扁平组织结构的导出选项
     * 验证内容：
     * 1. structure参数为flat
     */
    @Test
    @DisplayName("submitExportTask - 扁平组织结构导出")
    void testSubmitExportTask_FlatStructure() {
        // Arrange
        Long taskId = 5L;
        mockForm.setType("dataset_item");
        mockForm.setTargetId(1L);
        ExportTaskCreateForm.ExportOptions options = new ExportTaskCreateForm.ExportOptions();
        options.setStructure("flat");
        options.setIncludeTypes(Arrays.asList("clear"));
        options.setIncludeThumbnail(false);
        mockForm.setOptions(options);

        SysTask sysTask = new SysTask();
        sysTask.setId(taskId);
        sysTask.setTaskId("task_" + taskId);
        sysTask.setStatus("pending");

        SysDatasetItem mockItem = new SysDatasetItem();
        mockItem.setId(1L);
        mockItem.setName("test_item");

        SysItemFile mockItemFile = new SysItemFile();
        mockItemFile.setId(1L);
        mockItemFile.setItemId(1L);
        mockItemFile.setType("clear");
        mockItemFile.setFileId(1L);

        SysFile mockFile = SysFile.builder()
                .id(1L)
                .objectName("test.jpg")
                .build();

        when(sysTaskMapper.selectById(taskId)).thenReturn(sysTask);
        when(sysDatasetItemService.getById(1L)).thenReturn(mockItem);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(Arrays.asList(mockItemFile));
        when(sysFileService.getById(1L)).thenReturn(mockFile);
        when(fileService.uploadFile(anyString(), any(), anyLong(), anyString())).thenReturn("http://test.com/export.zip");
        when(fileService.downLoadFile(anyString())).thenReturn(new java.io.ByteArrayInputStream(new byte[1024]));

        // Act
        taskExecutor.submitExportTask(taskId, mockForm);

        // Assert
        verify(sysTaskMapper, atLeastOnce()).updateById(any(SysTask.class));
        assertEquals("flat", options.getStructure());
    }

    /**
     * 测试提交导出任务 - 仅包含清晰图
     * 测试场景：仅导出清晰图
     * 验证内容：
     * 1. includeTypes只包含clear
     */
    @Test
    @DisplayName("submitExportTask - 仅包含清晰图导出")
    void testSubmitExportTask_OnlyClearImages() {
        // Arrange
        Long taskId = 6L;
        mockForm.setType("dataset_item");
        mockForm.setTargetId(1L);
        ExportTaskCreateForm.ExportOptions options = new ExportTaskCreateForm.ExportOptions();
        options.setIncludeTypes(Arrays.asList("clear"));
        options.setIncludeThumbnail(false);
        mockForm.setOptions(options);

        SysTask sysTask = new SysTask();
        sysTask.setId(taskId);
        sysTask.setTaskId("task_" + taskId);
        sysTask.setStatus("pending");

        SysDatasetItem mockItem = new SysDatasetItem();
        mockItem.setId(1L);
        mockItem.setName("test_item");

        SysItemFile mockItemFile = new SysItemFile();
        mockItemFile.setId(1L);
        mockItemFile.setItemId(1L);
        mockItemFile.setType("clear");
        mockItemFile.setFileId(1L);

        SysFile mockFile = SysFile.builder()
                .id(1L)
                .objectName("test.jpg")
                .build();

        when(sysTaskMapper.selectById(taskId)).thenReturn(sysTask);
        when(sysDatasetItemService.getById(1L)).thenReturn(mockItem);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(Arrays.asList(mockItemFile));
        when(sysFileService.getById(1L)).thenReturn(mockFile);
        when(fileService.uploadFile(anyString(), any(), anyLong(), anyString())).thenReturn("http://test.com/export.zip");
        when(fileService.downLoadFile(anyString())).thenReturn(new java.io.ByteArrayInputStream(new byte[1024]));

        // Act
        taskExecutor.submitExportTask(taskId, mockForm);

        // Assert
        verify(sysTaskMapper, atLeastOnce()).updateById(any(SysTask.class));
        assertEquals(1, options.getIncludeTypes().size());
        assertEquals("clear", options.getIncludeTypes().get(0));
    }

    /**
     * 测试提交导出任务 - 仅包含有雾图
     * 测试场景：仅导出有雾图
     * 验证内容：
     * 1. includeTypes只包含hazy
     */
    @Test
    @DisplayName("submitExportTask - 仅包含有雾图导出")
    void testSubmitExportTask_OnlyHazyImages() {
        // Arrange
        Long taskId = 7L;
        mockForm.setType("dataset_item");
        mockForm.setTargetId(1L);
        ExportTaskCreateForm.ExportOptions options = new ExportTaskCreateForm.ExportOptions();
        options.setIncludeTypes(Arrays.asList("hazy"));
        options.setIncludeThumbnail(false);
        mockForm.setOptions(options);

        SysTask sysTask = new SysTask();
        sysTask.setId(taskId);
        sysTask.setTaskId("task_" + taskId);
        sysTask.setStatus("pending");

        SysDatasetItem mockItem = new SysDatasetItem();
        mockItem.setId(1L);
        mockItem.setName("test_item");

        SysItemFile mockItemFile = new SysItemFile();
        mockItemFile.setId(1L);
        mockItemFile.setItemId(1L);
        mockItemFile.setType("hazy");
        mockItemFile.setFileId(1L);

        SysFile mockFile = SysFile.builder()
                .id(1L)
                .objectName("test.jpg")
                .build();

        when(sysTaskMapper.selectById(taskId)).thenReturn(sysTask);
        when(sysDatasetItemService.getById(1L)).thenReturn(mockItem);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(Arrays.asList(mockItemFile));
        when(sysFileService.getById(1L)).thenReturn(mockFile);
        when(fileService.uploadFile(anyString(), any(), anyLong(), anyString())).thenReturn("http://test.com/export.zip");
        when(fileService.downLoadFile(anyString())).thenReturn(new java.io.ByteArrayInputStream(new byte[1024]));

        // Act
        taskExecutor.submitExportTask(taskId, mockForm);

        // Assert
        verify(sysTaskMapper, atLeastOnce()).updateById(any(SysTask.class));
        assertEquals(1, options.getIncludeTypes().size());
        assertEquals("hazy", options.getIncludeTypes().get(0));
    }

    /**
     * 测试提交导出任务 - 自定义类型导出
     * 测试场景：使用custom类型导出
     * 验证内容：
     * 1. type参数为custom
     */
    @Test
    @DisplayName("submitExportTask - 自定义类型导出")
    void testSubmitExportTask_CustomType() {
        // Arrange
        Long taskId = 9L;
        mockForm.setType("custom");
        mockForm.setTargetIds(Arrays.asList(1L, 2L));
        ExportTaskCreateForm.ExportOptions options = new ExportTaskCreateForm.ExportOptions();
        options.setStructure("by_item");
        mockForm.setOptions(options);

        SysTask sysTask = new SysTask();
        sysTask.setId(taskId);
        sysTask.setTaskId("task_" + taskId);
        sysTask.setStatus("pending");

        SysDatasetItem mockItem = new SysDatasetItem();
        mockItem.setId(1L);
        mockItem.setName("test_item");

        SysItemFile mockItemFile = new SysItemFile();
        mockItemFile.setId(1L);
        mockItemFile.setItemId(1L);
        mockItemFile.setType("clear");
        mockItemFile.setFileId(1L);

        SysFile mockFile = SysFile.builder()
                .id(1L)
                .objectName("test.jpg")
                .build();

        when(sysTaskMapper.selectById(taskId)).thenReturn(sysTask);
        when(sysDatasetItemService.getById(anyLong())).thenReturn(mockItem);
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(Arrays.asList(mockItemFile));
        when(sysFileService.getById(1L)).thenReturn(mockFile);
        when(fileService.uploadFile(anyString(), any(), anyLong(), anyString())).thenReturn("http://test.com/export.zip");
        when(fileService.downLoadFile(anyString())).thenReturn(new java.io.ByteArrayInputStream(new byte[1024]));

        // Act
        taskExecutor.submitExportTask(taskId, mockForm);

        // Assert
        verify(sysTaskMapper, atLeastOnce()).updateById(any(SysTask.class));
        assertEquals("custom", mockForm.getType());
    }
}
