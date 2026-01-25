package com.pei.dehaze.service;

import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.model.vo.TaskVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.impl.TaskServiceImpl;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.time.LocalDateTime;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * 任务服务单元测试
 * 测试目的：验证任务服务接口的调用正确性
 * 测试范围：
 * 1. 创建导出任务方法
 * 2. 查询任务状态方法
 * 3. 下载导出文件方法
 * 4. 取消任务方法
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("任务服务测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class TaskServiceTest {

    @Mock
    private SysTaskMapper sysTaskMapper;

    @Mock
    private TaskExecutor taskExecutor;

    @Mock
    private RedisTemplate<String, Object> redisTemplate;

    @Mock
    private ValueOperations<String, Object> valueOperations;

    private TaskServiceImpl taskService;

    private ExportTaskCreateForm mockForm;
    private TaskVO mockTaskVO;

    @BeforeEach
    void setUp() {
        mockForm = new ExportTaskCreateForm();
        mockForm.setType("dataset");
        mockForm.setTargetId(1L);

        mockTaskVO = new TaskVO();
        mockTaskVO.setTaskId("task-123");
        mockTaskVO.setStatus("pending");
        mockTaskVO.setProgress(0);
        mockTaskVO.setCreatedAt(LocalDateTime.now());

        // Mock redisTemplate behavior
        when(redisTemplate.opsForValue()).thenReturn(valueOperations);

        // 使用Spy创建TaskServiceImpl
        taskService = org.mockito.Mockito.spy(new TaskServiceImpl());

        // 手动注入依赖（非final字段）
        try {
            java.lang.reflect.Field taskExecutorField = taskService.getClass().getDeclaredField("taskExecutor");
            taskExecutorField.setAccessible(true);
            taskExecutorField.set(taskService, taskExecutor);

            java.lang.reflect.Field redisTemplateField = taskService.getClass().getDeclaredField("redisTemplate");
            redisTemplateField.setAccessible(true);
            redisTemplateField.set(taskService, redisTemplate);
        } catch (Exception e) {
            throw new RuntimeException("Failed to inject dependencies", e);
        }

        // Mock save方法，因为baseMapper无法注入
        org.mockito.Mockito.doAnswer(invocation -> {
            SysTask task = invocation.getArgument(0);
            if (task.getId() == null) {
                task.setId(1L);
            }
            return true;
        }).when(taskService).save(any(SysTask.class));

        // Mock updateById方法
        org.mockito.Mockito.doReturn(true).when(taskService).updateById(any(SysTask.class));
    }

    // ==================== 创建导出任务测试 ====================

    /**
     * 测试创建导出任务 - 单个数据集导出
     * 测试场景：创建单个数据集的导出任务
     * 验证内容：
     * 1. 方法被正确调用
     * 2. 返回任务VO对象
     * 注意：当前实现为空，返回null，待实现后验证返回值
     */
    @Test
    @DisplayName("createTask - 创建单个数据集导出任务")
    void testCreateTask_SingleDataset() {
        // Arrange
        mockForm.setType("dataset");
        mockForm.setTargetId(1L);
        mockForm.setTargetIds(null);

        try (MockedStatic<SecurityUtils> mockedSecurityUtils = mockStatic(SecurityUtils.class)) {
            mockedSecurityUtils.when(SecurityUtils::getUserId).thenReturn(1L);

            // Act
            TaskVO result = taskService.createTask(mockForm);

            // Assert
            assertNotNull(result);
            assertEquals("PENDING", result.getStatus());
            assertNotNull(result.getTaskId());
            assertEquals(0, result.getProgress());
        }
    }

    /**
     * 测试创建导出任务 - 批量数据集导出
     * 测试场景：创建多个数据集的批量导出任务
     * 验证内容：
     * 1. 方法被正确调用
     * 2. 批量导出参数正确传递
     */
    @Test
    @DisplayName("createTask - 创建批量数据集导出任务")
    void testCreateTask_BatchDatasets() {
        // Arrange
        mockForm.setType("batch_items");
        mockForm.setTargetId(null);
        mockForm.setTargetIds(java.util.Arrays.asList(1L, 2L, 3L));

        try (MockedStatic<SecurityUtils> mockedSecurityUtils = mockStatic(SecurityUtils.class)) {
            mockedSecurityUtils.when(SecurityUtils::getUserId).thenReturn(1L);

            // Act
            TaskVO result = taskService.createTask(mockForm);

            // Assert
            assertNotNull(result);
            assertEquals("PENDING", result.getStatus());
            assertNotNull(result.getTaskId());
        }
    }

    /**
     * 测试创建导出任务 - 包含导出选项
     * 测试场景：创建任务时指定导出选项
     * 验证内容：
     * 1. 导出选项参数正确传递
     * 2. 选项值正确设置
     */
    @Test
    @DisplayName("createTask - 包含导出选项")
    void testCreateTask_WithOptions() {
        // Arrange
        mockForm.setType("dataset");
        mockForm.setTargetId(1L);
        ExportTaskCreateForm.ExportOptions options = new ExportTaskCreateForm.ExportOptions();
        options.setStructure("flat");
        options.setIncludeTypes(java.util.Arrays.asList("clear", "hazy"));
        options.setIncludeThumbnail(true);
        mockForm.setOptions(options);

        try (MockedStatic<SecurityUtils> mockedSecurityUtils = mockStatic(SecurityUtils.class)) {
            mockedSecurityUtils.when(SecurityUtils::getUserId).thenReturn(1L);

            // Act
            TaskVO result = taskService.createTask(mockForm);

            // Assert
            assertNotNull(result);
            assertEquals("PENDING", result.getStatus());
            assertNotNull(result.getTaskId());
        }
    }

    // ==================== 查询任务状态测试 ====================

    /**
     * 测试查询任务状态 - pending状态
     * 测试场景：查询pending状态的任务
     * 验证内容：
     * 1. 返回任务VO对象
     * 2. 状态为PENDING
     */
    @Test
    @DisplayName("getTaskStatus - 查询pending状态任务")
    void testGetTaskStatus_Pending() {
        // Arrange
        String taskId = "task-123";

        SysTask sysTask = new SysTask();
        sysTask.setTaskId(taskId);
        sysTask.setStatus("PENDING");
        sysTask.setProgress(0);

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act
        TaskVO result = taskService.getTaskStatus(taskId);

        // Assert
        assertNotNull(result);
        assertEquals(taskId, result.getTaskId());
        assertEquals("PENDING", result.getStatus());
    }

    /**
     * 测试查询任务状态 - processing状态
     * 测试场景：查询processing状态的任务
     * 验证内容：
     * 1. 返回任务VO对象
     * 2. 状态为PROCESSING，进度信息正确
     */
    @Test
    @DisplayName("getTaskStatus - 查询processing状态任务")
    void testGetTaskStatus_Processing() {
        // Arrange
        String taskId = "task-456";

        SysTask sysTask = new SysTask();
        sysTask.setTaskId(taskId);
        sysTask.setStatus("PROCESSING");
        sysTask.setProgress(50);
        sysTask.setTotalFiles(100);
        sysTask.setProcessedFiles(50);

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act
        TaskVO result = taskService.getTaskStatus(taskId);

        // Assert
        assertNotNull(result);
        assertEquals(taskId, result.getTaskId());
        assertEquals("PROCESSING", result.getStatus());
        assertEquals(50, result.getProgress());
    }

    /**
     * 测试查询任务状态 - completed状态
     * 测试场景：查询completed状态的任务
     * 验证内容：
     * 1. 返回任务VO对象
     * 2. 状态为COMPLETED，包含下载链接
     */
    @Test
    @DisplayName("getTaskStatus - 查询completed状态任务")
    void testGetTaskStatus_Completed() {
        // Arrange
        String taskId = "task-789";

        SysTask sysTask = new SysTask();
        sysTask.setTaskId(taskId);
        sysTask.setStatus("COMPLETED");
        sysTask.setProgress(100);
        sysTask.setResult("http://test.com/export.zip");

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act
        TaskVO result = taskService.getTaskStatus(taskId);

        // Assert
        assertNotNull(result);
        assertEquals(taskId, result.getTaskId());
        assertEquals("COMPLETED", result.getStatus());
        assertEquals("http://test.com/export.zip", result.getDownloadUrl());
    }

    /**
     * 测试查询任务状态 - failed状态
     * 测试场景：查询failed状态的任务
     * 验证内容：
     * 1. 返回任务VO对象
     * 2. 状态为FAILED，包含错误信息
     */
    @Test
    @DisplayName("getTaskStatus - 查询failed状态任务")
    void testGetTaskStatus_Failed() {
        // Arrange
        String taskId = "task-failed";

        SysTask sysTask = new SysTask();
        sysTask.setTaskId(taskId);
        sysTask.setStatus("FAILED");
        sysTask.setErrorMessage("导出失败：文件不存在");

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act
        TaskVO result = taskService.getTaskStatus(taskId);

        // Assert
        assertNotNull(result);
        assertEquals(taskId, result.getTaskId());
        assertEquals("FAILED", result.getStatus());
        assertEquals("导出失败：文件不存在", result.getError());
    }

    // ==================== 下载导出文件测试 ====================

    /**
     * 测试下载导出文件 - 成功场景
     * 测试场景：下载已完成的导出任务文件
     * 验证内容：
     * 1. 返回有效的下载链接
     */
    @Test
    @DisplayName("getDownloadUrl - 下载已完成任务文件")
    void testGetDownloadUrl_Success() {
        // Arrange
        String taskId = "task-completed";

        SysTask sysTask = new SysTask();
        sysTask.setTaskId(taskId);
        sysTask.setStatus("COMPLETED");
        sysTask.setResult("http://test.com/export.zip");
        sysTask.setExpiresAt(LocalDateTime.now().plusHours(1));

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act
        String result = taskService.getDownloadUrl(taskId);

        // Assert
        assertNotNull(result);
        assertEquals("http://test.com/export.zip", result);
    }

    /**
     * 测试下载导出文件 - 未完成状态
     * 测试场景：尝试下载未完成的任务文件
     * 验证内容：
     * 1. 返回null
     */
    @Test
    @DisplayName("getDownloadUrl - 未完成任务不可下载")
    void testGetDownloadUrl_NotCompleted() {
        // Arrange
        String taskId = "task-pending";

        SysTask sysTask = new SysTask();
        sysTask.setTaskId(taskId);
        sysTask.setStatus("PENDING");

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act
        String result = taskService.getDownloadUrl(taskId);

        // Assert
        assertNull(result);
    }

    /**
     * 测试下载导出文件 - 已过期
     * 测试场景：下载已过期的任务文件
     * 验证内容：
     * 1. 返回null
     */
    @Test
    @DisplayName("getDownloadUrl - 已过期任务不可下载")
    void testGetDownloadUrl_Expired() {
        // Arrange
        String taskId = "task-expired";

        SysTask sysTask = new SysTask();
        sysTask.setTaskId(taskId);
        sysTask.setStatus("COMPLETED");
        sysTask.setResult("http://test.com/export.zip");
        sysTask.setExpiresAt(LocalDateTime.now().minusHours(1));

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act
        String result = taskService.getDownloadUrl(taskId);

        // Assert
        assertNull(result);
    }

    // ==================== 取消任务测试 ====================

    /**
     * 测试取消任务 - 等待中状态
     * 测试场景：取消等待中的任务
     * 验证内容：
     * 1. 任务状态更新为CANCELLED
     */
    @Test
    @DisplayName("cancelTask - 取消等待中的任务")
    void testCancelTask_Pending() {
        // Arrange
        String taskId = "task-pending";

        SysTask sysTask = new SysTask();
        sysTask.setId(1L);
        sysTask.setTaskId(taskId);
        sysTask.setStatus("PENDING");

        when(valueOperations.get(anyString())).thenReturn(sysTask);
        doNothing().when(valueOperations).set(anyString(), any(), anyLong(), any());

        // Act
        taskService.cancelTask(taskId);

        // Assert
        verify(valueOperations, atLeastOnce()).set(eq("task:" + taskId), any(), anyLong(), any());
    }

    /**
     * 测试取消任务 - 处理中状态
     * 测试场景：取消处理中的任务
     * 验证内容：
     * 1. 任务状态更新为CANCELLED
     */
    @Test
    @DisplayName("cancelTask - 取消处理中的任务")
    void testCancelTask_Processing() {
        // Arrange
        String taskId = "task-processing";

        SysTask sysTask = new SysTask();
        sysTask.setId(2L);
        sysTask.setTaskId(taskId);
        sysTask.setStatus("PROCESSING");

        when(valueOperations.get(anyString())).thenReturn(sysTask);
        doNothing().when(valueOperations).set(anyString(), any(), anyLong(), any());

        // Act
        taskService.cancelTask(taskId);

        // Assert
        verify(valueOperations, atLeastOnce()).set(eq("task:" + taskId), any(), anyLong(), any());
    }

    /**
     * 测试取消任务 - 已完成状态
     * 测试场景：尝试取消已完成的任务
     * 验证内容：
     * 1. 不更新任务状态
     */
    @Test
    @DisplayName("cancelTask - 已完成任务不可取消")
    void testCancelTask_Completed() {
        // Arrange
        String taskId = "task-completed";

        SysTask sysTask = new SysTask();
        sysTask.setId(3L);
        sysTask.setTaskId(taskId);
        sysTask.setStatus("COMPLETED");

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act
        taskService.cancelTask(taskId);

        // Assert
        verify(sysTaskMapper, never()).updateById(any());
    }

    /**
     * 测试取消任务 - 任务不存在
     * 测试场景：取消不存在的任务
     * 验证内容：
     * 1. 抛出异常
     */
    @Test
    @DisplayName("cancelTask - 任务不存在")
    void testCancelTask_NotFound() {
        // Arrange
        String taskId = "task-not-found";

        when(valueOperations.get(anyString())).thenReturn(null);

        // Act & Assert
        assertThrows(Exception.class, () -> taskService.cancelTask(taskId));
    }
}
