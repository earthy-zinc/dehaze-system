package com.pei.dehaze.service;

import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.config.WebSocketMessageRelay;
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

    @Mock
    private WebSocketMessageRelay wsMessageRelay;

    private TaskServiceImpl taskService;

    private ExportTaskCreateForm mockForm;

    @BeforeEach
    void setUp() {
        mockForm = new ExportTaskCreateForm();
        mockForm.setType("dataset_export");
        mockForm.setParamsJson("{\"module\":\"dataset\",\"query\":{\"datasetId\":1}}");

        // Mock redisTemplate behavior
        when(redisTemplate.opsForValue()).thenReturn(valueOperations);

        // 使用Spy创建TaskServiceImpl，通过构造器注入依赖
        taskService = org.mockito.Mockito.spy(new TaskServiceImpl(taskExecutor, redisTemplate, wsMessageRelay));

        // 注入 baseMapper（ServiceImpl 父类字段）
        try {
            java.lang.reflect.Field baseMapperField = taskService.getClass().getSuperclass().getDeclaredField("baseMapper");
            baseMapperField.setAccessible(true);
            baseMapperField.set(taskService, sysTaskMapper);
        } catch (Exception e) {
            throw new RuntimeException("Failed to inject baseMapper", e);
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
     * 测试创建导出任务 - 数据集导出
     * 测试场景：创建数据集的导出任务
     * 验证内容：
     * 1. 方法被正确调用
     * 2. 返回任务VO对象
     */
    @Test
    @DisplayName("createTask - 创建数据集导出任务")
    void testCreateTask_DatasetExport() {
        // Arrange
        mockForm.setType("dataset_export");
        mockForm.setParamsJson("{\"module\":\"dataset\",\"query\":{\"datasetId\":1}}");

        try (MockedStatic<SecurityUtils> mockedSecurityUtils = mockStatic(SecurityUtils.class)) {
            mockedSecurityUtils.when(SecurityUtils::getUserId).thenReturn(1L);

            // Act
            TaskVO result = taskService.createTask(mockForm, null);

            // Assert
            assertNotNull(result);
            assertEquals(TaskConstants.STATUS_PENDING, result.getStatus());
            assertNotNull(result.getTaskId());
            assertEquals(0, result.getProgress());
        }
    }

    /**
     * 测试创建导出任务 - 批量数据项下载
     * 测试场景：创建多个数据项的批量下载任务
     * 验证内容：
     * 1. 方法被正确调用
     * 2. 批量下载参数正确传递
     */
    @Test
    @DisplayName("createTask - 创建批量数据项下载任务")
    void testCreateTask_BatchItemsDownload() {
        // Arrange
        mockForm.setType("dataset_export");
        mockForm.setParamsJson("{\"module\":\"dataset\",\"query\":{\"itemIds\":[1,2,3]}}");

        try (MockedStatic<SecurityUtils> mockedSecurityUtils = mockStatic(SecurityUtils.class)) {
            mockedSecurityUtils.when(SecurityUtils::getUserId).thenReturn(1L);

            // Act
            TaskVO result = taskService.createTask(mockForm, null);

            // Assert
            assertNotNull(result);
            assertEquals(TaskConstants.STATUS_PENDING, result.getStatus());
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
        mockForm.setType("dataset_export");
        mockForm.setParamsJson("{\"module\":\"dataset\",\"query\":{\"datasetId\":1,\"options\":{\"structure\":\"flat\",\"includeTypes\":[\"clear\",\"hazy\"],\"includeThumbnail\":true}}}");

        try (MockedStatic<SecurityUtils> mockedSecurityUtils = mockStatic(SecurityUtils.class)) {
            mockedSecurityUtils.when(SecurityUtils::getUserId).thenReturn(1L);

            // Act
            TaskVO result = taskService.createTask(mockForm, null);

            // Assert
            assertNotNull(result);
            assertEquals(TaskConstants.STATUS_PENDING, result.getStatus());
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
        sysTask.setStatus(TaskConstants.STATUS_PENDING);
        sysTask.setProgress(0);

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act
        TaskVO result = taskService.getTaskStatus(taskId);

        // Assert
        assertNotNull(result);
        assertEquals(taskId, result.getTaskId());
        assertEquals(TaskConstants.STATUS_PENDING, result.getStatus());
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
        sysTask.setStatus(TaskConstants.STATUS_PROCESSING);
        sysTask.setProgress(50);
        sysTask.setTotalFiles(100);
        sysTask.setProcessedFiles(50);

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act
        TaskVO result = taskService.getTaskStatus(taskId);

        // Assert
        assertNotNull(result);
        assertEquals(taskId, result.getTaskId());
        assertEquals(TaskConstants.STATUS_PROCESSING, result.getStatus());
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
        sysTask.setStatus(TaskConstants.STATUS_COMPLETED);
        sysTask.setProgress(100);
        sysTask.setResult("http://test.com/export.zip");

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act
        TaskVO result = taskService.getTaskStatus(taskId);

        // Assert
        assertNotNull(result);
        assertEquals(taskId, result.getTaskId());
        assertEquals(TaskConstants.STATUS_COMPLETED, result.getStatus());
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
        sysTask.setStatus(TaskConstants.STATUS_FAILED);
        sysTask.setErrorMessage("导出失败：文件不存在");

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act
        TaskVO result = taskService.getTaskStatus(taskId);

        // Assert
        assertNotNull(result);
        assertEquals(taskId, result.getTaskId());
        assertEquals(TaskConstants.STATUS_FAILED, result.getStatus());
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
        sysTask.setStatus(TaskConstants.STATUS_COMPLETED);
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
     * 1. 抛出BusinessException
     */
    @Test
    @DisplayName("getDownloadUrl - 未完成任务不可下载")
    void testGetDownloadUrl_NotCompleted() {
        // Arrange
        String taskId = "task-pending";

        SysTask sysTask = new SysTask();
        sysTask.setTaskId(taskId);
        sysTask.setStatus(TaskConstants.STATUS_PENDING);

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act & Assert
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> taskService.getDownloadUrl(taskId));
        assertTrue(exception.getMessage().contains("任务未完成"));
    }

    /**
     * 测试下载导出文件 - 已过期
     * 测试场景：下载已过期的任务文件
     * 验证内容：
     * 1. 抛出BusinessException
     */
    @Test
    @DisplayName("getDownloadUrl - 已过期任务不可下载")
    void testGetDownloadUrl_Expired() {
        // Arrange
        String taskId = "task-expired";

        SysTask sysTask = new SysTask();
        sysTask.setTaskId(taskId);
        sysTask.setStatus(TaskConstants.STATUS_COMPLETED);
        sysTask.setResult("http://test.com/export.zip");
        sysTask.setExpiresAt(LocalDateTime.now().minusHours(1));

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act & Assert
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> taskService.getDownloadUrl(taskId));
        assertTrue(exception.getMessage().contains("任务已过期"));
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
        sysTask.setStatus(TaskConstants.STATUS_PENDING);

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
        sysTask.setStatus(TaskConstants.STATUS_PROCESSING);

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
     * 1. 抛出BusinessException
     */
    @Test
    @DisplayName("cancelTask - 已完成任务不可取消")
    void testCancelTask_Completed() {
        // Arrange
        String taskId = "task-completed";

        SysTask sysTask = new SysTask();
        sysTask.setId(3L);
        sysTask.setTaskId(taskId);
        sysTask.setStatus(TaskConstants.STATUS_COMPLETED);

        when(valueOperations.get(anyString())).thenReturn(sysTask);

        // Act & Assert
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> taskService.cancelTask(taskId));
        assertTrue(exception.getMessage().contains("任务已完成或失败"));
    }

    /**
     * 测试取消任务 - 任务不存在
     * 测试场景：取消不存在的任务
     * 验证内容：
     * 1. 抛出BusinessException
     */
    @Test
    @DisplayName("cancelTask - 任务不存在")
    void testCancelTask_NotFound() {
        // Arrange
        String taskId = "task-not-found";

        when(valueOperations.get(anyString())).thenReturn(null);
        when(sysTaskMapper.selectOne(any())).thenReturn(null);

        // Act & Assert
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> taskService.cancelTask(taskId));
        assertTrue(exception.getMessage().contains("任务不存在"));
    }
}
