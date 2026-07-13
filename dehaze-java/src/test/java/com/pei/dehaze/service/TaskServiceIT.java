package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.vo.TaskVO;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 任务服务集成测试
 * 测试目的：验证导出任务的完整业务流程
 * 测试范围：任务创建、状态查询、下载链接生成、任务取消
 */
@SpringBootTest
@DisplayName("任务服务集成测试")
class TaskServiceIT {

    @Autowired
    private TaskService taskService;

    @Autowired
    private SysTaskMapper taskMapper;

    private String testTaskId;

    @BeforeEach
    void setUp() {
        testTaskId = UUID.randomUUID().toString();
    }

    /**
     * 辅助方法：根据taskId查询任务
     */
    private SysTask selectByTaskId(String taskId) {
        LambdaQueryWrapper<SysTask> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(SysTask::getTaskId, taskId);
        return taskMapper.selectOne(queryWrapper);
    }

    /**
     * 测试目的：验证创建导出任务
     * 测试场景：创建一个新的导出任务
     * 验证内容：任务成功创建，返回任务ID
     */
    @Test
    @DisplayName("创建导出任务-成功")
    @Transactional
    void testCreateExportTask_Success() {
        // 创建测试任务记录
        SysTask task = new SysTask();
        task.setTaskId(testTaskId);
        task.setTaskType("DATASET_EXPORT");
        task.setStatus("pending");
        task.setProgress(0);
        task.setTotalFiles(0);
        task.setProcessedFiles(0);
        task.setCreateTime(LocalDateTime.now());

        taskMapper.insert(task);

        // 验证任务已创建
        SysTask fromDb = selectByTaskId(testTaskId);
        assertNotNull(fromDb);
        assertEquals(testTaskId, fromDb.getTaskId());
        assertEquals("pending", fromDb.getStatus());
    }

    /**
     * 测试目的：验证查询任务状态
     * 测试场景：查询已存在的任务状态
     * 验证内容：返回正确的任务状态信息
     */
    @Test
    @DisplayName("查询任务状态-成功")
    @Transactional
    void testGetTaskStatus_Success() {
        // 创建测试任务
        SysTask task = new SysTask();
        task.setTaskId(testTaskId);
        task.setTaskType("DATASET_EXPORT");
        task.setStatus("processing");
        task.setProgress(50);
        task.setTotalFiles(100);
        task.setProcessedFiles(50);
        task.setCreateTime(LocalDateTime.now());
        taskMapper.insert(task);

        // 查询任务状态
        TaskVO result = taskService.getTaskStatus(testTaskId);

        assertNotNull(result);
        assertEquals(testTaskId, result.getTaskId());
        assertEquals("processing", result.getStatus());
        assertEquals(50, result.getProgress());
    }

    /**
     * 测试目的：验证查询不存在的任务状态
     * 测试场景：查询不存在的任务ID
     * 验证内容：返回null
     */
    @Test
    @DisplayName("查询不存在的任务状态-返回null")
    void testGetTaskStatus_NotFound() {
        String nonExistentTaskId = "non-existent-task-id-12345";

        TaskVO result = taskService.getTaskStatus(nonExistentTaskId);

        assertNull(result);
    }

    /**
     * 测试目的：验证查询任务状态时为空任务ID的异常处理
     * 测试场景：传入空的任务ID
     * 验证内容：抛出BusinessException异常
     */
    @Test
    @DisplayName("查询任务状态-空任务ID抛出异常")
    void testGetTaskStatus_NullTaskId() {
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> taskService.getTaskStatus(""));

        assertTrue(exception.getMessage().contains("任务ID不能为空"));
    }

    /**
     * 测试目的：验证任务完成后的状态
     * 测试场景：更新任务为完成状态
     * 验证内容：任务状态和结果正确更新
     */
    @Test
    @DisplayName("任务完成状态测试-成功")
    @Transactional
    void testTaskCompleted_Success() {
        // 创建测试任务
        SysTask task = new SysTask();
        task.setTaskId(testTaskId);
        task.setTaskType("DATASET_EXPORT");
        task.setStatus("completed");
        task.setProgress(100);
        task.setTotalFiles(100);
        task.setProcessedFiles(100);
        task.setResult("http://example.com/download/file.zip");
        task.setCreateTime(LocalDateTime.now());
        task.setStartedAt(LocalDateTime.now().minusMinutes(5));
        task.setCompletedAt(LocalDateTime.now());
        taskMapper.insert(task);

        // 查询任务状态
        TaskVO result = taskService.getTaskStatus(testTaskId);

        assertNotNull(result);
        assertEquals("completed", result.getStatus());
        assertEquals(100, result.getProgress());
        assertNotNull(result.getDownloadUrl());
        assertTrue(result.getDownloadUrl().contains("download"));
    }

    /**
     * 测试目的：验证任务失败状态
     * 测试场景：任务执行失败
     * 验证内容：任务状态和错误信息正确记录
     */
    @Test
    @DisplayName("任务失败状态测试-成功")
    @Transactional
    void testTaskFailed_Success() {
        // 创建失败的任务
        SysTask task = new SysTask();
        task.setTaskId(testTaskId);
        task.setTaskType("DATASET_EXPORT");
        task.setStatus("failed");
        task.setProgress(60);
        task.setTotalFiles(100);
        task.setProcessedFiles(60);
        task.setErrorMessage("文件处理失败：磁盘空间不足");
        task.setCreateTime(LocalDateTime.now());
        task.setStartedAt(LocalDateTime.now().minusMinutes(3));
        taskMapper.insert(task);

        // 查询任务状态
        TaskVO result = taskService.getTaskStatus(testTaskId);

        assertNotNull(result);
        assertEquals("failed", result.getStatus());
        assertEquals(60, result.getProgress());
        assertNotNull(result.getError());
        assertTrue(result.getError().contains("失败"));
    }

    /**
     * 测试目的：验证取消任务
     * 测试场景：取消一个正在执行的任务
     * 验证内容：任务状态更新为cancelled
     */
    @Test
    @DisplayName("取消任务-成功")
    @Transactional
    void testCancelTask_Success() {
        // 创建正在执行的任务
        SysTask task = new SysTask();
        task.setTaskId(testTaskId);
        task.setTaskType("DATASET_EXPORT");
        task.setStatus("processing");
        task.setProgress(30);
        task.setTotalFiles(100);
        task.setProcessedFiles(30);
        task.setCreateTime(LocalDateTime.now());
        task.setStartedAt(LocalDateTime.now().minusMinutes(1));
        taskMapper.insert(task);

        // 取消任务
        taskService.cancelTask(testTaskId);

        // 验证任务状态
        SysTask fromDb = selectByTaskId(testTaskId);
        assertNotNull(fromDb);
        assertEquals("cancelled", fromDb.getStatus());
    }

    /**
     * 测试目的：验证下载不存在任务的处理
     * 测试场景：下载不存在的任务
     * 验证内容：抛出IllegalArgumentException异常
     */
    @Test
    @DisplayName("下载不存在任务-抛出异常")
    void testGetDownloadUrl_NotFound() {
        String nonExistentTaskId = "non-existent-task-id-12345";

        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> taskService.getDownloadUrl(nonExistentTaskId));

        assertTrue(exception.getMessage().contains("任务不存在"));
    }

    /**
     * 测试目的：验证下载任务时为空任务ID的异常处理
     * 测试场景：传入空的任务ID
     * 验证内容：抛出BusinessException异常
     */
    @Test
    @DisplayName("下载任务-空任务ID抛出异常")
    void testGetDownloadUrl_NullTaskId() {
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> taskService.getDownloadUrl(""));

        assertTrue(exception.getMessage().contains("任务ID不能为空"));
    }

    /**
     * 测试目的：验证取消任务时为空任务ID的异常处理
     * 测试场景：传入空的任务ID
     * 验证内容：抛出BusinessException异常
     */
    @Test
    @DisplayName("取消任务-空任务ID抛出异常")
    void testCancelTask_NullTaskId() {
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> taskService.cancelTask(""));

        assertTrue(exception.getMessage().contains("任务ID不能为空"));
    }

    /**
     * 测试目的：验证取消不存在的任务
     * 测试场景：取消不存在的任务
     * 验证内容：抛出IllegalArgumentException异常
     */
    @Test
    @DisplayName("取消不存在的任务-抛出异常")
    void testCancelTask_NotFound() {
        String nonExistentTaskId = "non-existent-task-id-12345";

        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> taskService.cancelTask(nonExistentTaskId));

        assertTrue(exception.getMessage().contains("任务不存在"));
    }

    /**
     * 测试目的：验证任务进度更新
     * 测试场景：模拟任务执行过程中进度更新
     * 验证内容：进度正确递增
     */
    @Test
    @DisplayName("任务进度更新测试-成功")
    @Transactional
    void testTaskProgressUpdate_Success() {
        // 创建任务
        SysTask task = new SysTask();
        task.setTaskId(testTaskId);
        task.setTaskType("DATASET_EXPORT");
        task.setStatus("processing");
        task.setProgress(0);
        task.setTotalFiles(100);
        task.setProcessedFiles(0);
        task.setCreateTime(LocalDateTime.now());
        task.setStartedAt(LocalDateTime.now());
        taskMapper.insert(task);

        // 模拟进度更新
        task.setProgress(50);
        task.setProcessedFiles(50);
        taskMapper.updateById(task);

        // 验证进度
        SysTask updated = selectByTaskId(testTaskId);
        assertNotNull(updated);
        assertEquals(50, updated.getProgress());
        assertEquals(50, updated.getProcessedFiles());
    }
}
