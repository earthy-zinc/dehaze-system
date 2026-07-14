package com.pei.dehaze.service.strategy;

import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
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

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * 默认进度回调单元测试
 * 测试目的：验证进度更新的节流逻辑、取消检测和数据库更新
 * 测试范围：
 * 1. 进度更新节流（5%变化或2秒间隔）
 * 2. 任务取消检测
 * 3. 进度为100%时强制更新
 * 4. checkCancelled()异常抛出
 *
 * @author earthy-zinc
 * @since 2026-01-20
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("默认进度回调测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class DefaultProgressCallbackTest {

    @Mock
    private SysTaskMapper taskMapper;

    @Mock
    private RedisTemplate<String, Object> redisTemplate;

    @Mock
    private ValueOperations<String, Object> valueOperations;

    @InjectMocks
    private DefaultProgressCallback progressCallback;

    @BeforeEach
    void setUp() {
        when(redisTemplate.opsForValue()).thenReturn(valueOperations);

        progressCallback = new DefaultProgressCallback(1L, "task-1", 1L, taskMapper, redisTemplate, null);

        when(valueOperations.get(anyString())).thenReturn(false);
    }

    // ==================== 进度更新测试 ====================

    /**
     * 测试进度更新 - 基本场景
     * 测试场景：首次调用updateProgress，进度为50%
     * 验证内容：
     * 1. 数据库updateById被调用
     * 2. 进度正确计算
     * 3. processedFiles和totalFiles正确更新
     */
    @Test
    @DisplayName("updateProgress - 首次更新50%进度")
    void testUpdateProgress_FirstUpdate() {
        // Arrange
        when(taskMapper.updateById(any(SysTask.class))).thenReturn(1);

        // Act
        progressCallback.updateProgress(50, 100, "处理中");

        // Assert
        verify(taskMapper, times(1)).updateById(argThat(task -> {
            assertEquals(1L, task.getId());
            assertEquals(50, task.getProgress());
            assertEquals(50, task.getProcessedFiles());
            assertEquals(100, task.getTotalFiles());
            return true;
        }));
    }

    /**
     * 测试进度更新 - 进度变化>=5%触发更新
     * 测试场景：进度从0%变到6%（变化6%>5%）
     * 验证内容：
     * 1. 数据库updateById被调用
     */
    @Test
    @DisplayName("updateProgress - 进度变化5%以上触发更新")
    void testUpdateProgress_ChangeGreaterThan5Percent() {
        // Arrange
        when(taskMapper.updateById(any(SysTask.class))).thenReturn(1);

        progressCallback.updateProgress(0, 100, "开始");

        reset(taskMapper);

        // Act - 变化6% (6/100 * 100 = 6%)
        progressCallback.updateProgress(6, 100, "进行中");

        // Assert
        verify(taskMapper, times(1)).updateById(argThat(task ->
                task.getProgress() == 6
        ));
    }

    /**
     * 测试进度更新 - 进度变化<5%不触发更新
     * 测试场景：进度从0%变到4%（变化4%<5%）
     * 验证内容：
     * 1. 数据库updateById不被调用
     */
    @Test
    @DisplayName("updateProgress - 进度变化小于5%不触发更新")
    void testUpdateProgress_ChangeLessThan5Percent() {
        // Arrange
        when(taskMapper.updateById(any(SysTask.class))).thenReturn(1);

        progressCallback.updateProgress(0, 100, "开始");

        reset(taskMapper);

        // Act - 变化4% (4/100 * 100 = 4%)
        progressCallback.updateProgress(4, 100, "进行中");

        // Assert
        verify(taskMapper, never()).updateById(any(SysTask.class));
    }

    /**
     * 测试进度更新 - 正好5%变化触发更新
     * 测试场景：进度从0%变到5%（变化5%）
     * 验证内容：
     * 1. 数据库updateById被调用
     */
    @Test
    @DisplayName("updateProgress - 进度变化正好5%触发更新")
    void testUpdateProgress_ChangeExactly5Percent() {
        // Arrange
        when(taskMapper.updateById(any(SysTask.class))).thenReturn(1);

        progressCallback.updateProgress(0, 100, "开始");

        reset(taskMapper);

        // Act - 变化5% (5/100 * 100 = 5%)
        progressCallback.updateProgress(5, 100, "进行中");

        // Assert
        verify(taskMapper, times(1)).updateById(argThat(task ->
                task.getProgress() == 5
        ));
    }

    /**
     * 测试进度更新 - 100%进度强制更新
     * 测试场景：进度达到100%，即使变化小于5%也强制更新
     * 验证内容：
     * 1. 数据库updateById被调用
     */
    @Test
    @DisplayName("updateProgress - 进度100%强制更新")
    void testUpdateProgress_ForceUpdateOnComplete() {
        // Arrange
        when(taskMapper.updateById(any(SysTask.class))).thenReturn(1);

        progressCallback.updateProgress(99, 100, "即将完成");

        reset(taskMapper);

        // Act - 100%强制更新
        progressCallback.updateProgress(100, 100, "完成");

        // Assert
        verify(taskMapper, times(1)).updateById(argThat(task ->
                task.getProgress() == 100
        ));
    }

    /**
     * 测试进度更新 - 分母为0时进度为100%
     * 测试场景：total参数为0，进度应设为100%
     * 验证内容：
     * 1. 进度设为100
     * 2. 数据库updateById被调用
     */
    @Test
    @DisplayName("updateProgress - 分母为0时进度为100%")
    void testUpdateProgress_TotalZero() {
        // Arrange
        when(taskMapper.updateById(any(SysTask.class))).thenReturn(1);

        // Act
        progressCallback.updateProgress(0, 0, "无需处理");

        // Assert
        verify(taskMapper, times(1)).updateById(argThat(task ->
                task.getProgress() == 100
        ));
    }

    /**
     * 测试进度更新 - 时间间隔触发更新
     * 测试场景：进度变化小于5%，但时间间隔超过2秒
     * 验证内容：
     * 1. 数据库updateById被调用
     * 2. 由于时间间隔超过阈值，即使进度变化小也更新
     */
    @Test
    @DisplayName("updateProgress - 时间间隔超过2秒触发更新")
    void testUpdateProgress_TimeIntervalTrigger() throws InterruptedException {
        // Arrange
        when(taskMapper.updateById(any(SysTask.class))).thenReturn(1);

        progressCallback.updateProgress(1, 100, "第一步");

        reset(taskMapper);

        // 等待超过2秒
        Thread.sleep(2100);

        // Act - 进度只变化1% (2-1=1% < 5%)，但时间间隔>2秒
        progressCallback.updateProgress(2, 100, "第二步");

        // Assert
        verify(taskMapper, times(1)).updateById(argThat(task ->
                task.getProgress() == 2
        ));
    }

    // ==================== 任务取消检测测试 ====================

    /**
     * 测试取消检测 - 任务未取消
     * 测试场景：Redis中没有取消标记
     * 验证内容：
     * 1. isCancelled()返回false
     */
    @Test
    @DisplayName("isCancelled - 任务未取消")
    void testIsCancelled_NotCancelled() {
        // Arrange
        when(valueOperations.get(TaskConstants.TASK_CANCEL_PREFIX + "task-1")).thenReturn(false);

        // Act
        boolean cancelled = progressCallback.isCancelled();

        // Assert
        assertFalse(cancelled);
    }

    /**
     * 测试取消检测 - 任务已取消
     * 测试场景：Redis中有取消标记true
     * 验证内容：
     * 1. isCancelled()返回true
     */
    @Test
    @DisplayName("isCancelled - 任务已取消")
    void testIsCancelled_Cancelled() {
        // Arrange
        when(valueOperations.get(TaskConstants.TASK_CANCEL_PREFIX + "task-1")).thenReturn(true);

        // Act
        boolean cancelled = progressCallback.isCancelled();

        // Assert
        assertTrue(cancelled);
    }

    /**
     * 测试取消检测 - Redis返回null
     * 测试场景：Redis中没有对应的键
     * 验证内容：
     * 1. isCancelled()返回false
     */
    @Test
    @DisplayName("isCancelled - Redis返回null视为未取消")
    void testIsCancelled_NullValue() {
        // Arrange
        when(valueOperations.get(TaskConstants.TASK_CANCEL_PREFIX + "task-1")).thenReturn(null);

        // Act
        boolean cancelled = progressCallback.isCancelled();

        // Assert
        assertFalse(cancelled);
    }

    /**
     * 测试进度更新时检查取消 - 正常情况
     * 测试场景：调用updateProgress，任务未取消
     * 验证内容：
     * 1. 进度正常更新
     * 2. 不抛出异常
     */
    @Test
    @DisplayName("updateProgress - 未取消时正常更新")
    void testUpdateProgress_CheckCancelled_NotCancelled() {
        // Arrange
        when(taskMapper.updateById(any(SysTask.class))).thenReturn(1);

        // Act & Assert
        assertDoesNotThrow(() -> progressCallback.updateProgress(50, 100, "处理中"));

        verify(taskMapper, times(1)).updateById(any(SysTask.class));
    }

    /**
     * 测试进度更新时检查取消 - 任务已取消
     * 测试场景：调用updateProgress，任务已取消
     * 验证内容：
     * 1. 抛出TaskCancelledException异常
     * 2. 数据库不更新
     */
    @Test
    @DisplayName("updateProgress - 已取消时抛出异常")
    void testUpdateProgress_CheckCancelled_Cancelled() {
        // Arrange
        when(valueOperations.get(TaskConstants.TASK_CANCEL_PREFIX + "task-1")).thenReturn(true);
        when(taskMapper.updateById(any(SysTask.class))).thenReturn(1);

        // Act & Assert
        assertThrows(TaskCancelledException.class, () ->
                progressCallback.updateProgress(50, 100, "处理中")
        );

        verify(taskMapper, never()).updateById(any(SysTask.class));
    }

    // ==================== 边界场景测试 ====================

    /**
     * 测试进度更新 - current大于total
     * 测试场景：当前进度超过总数
     * 验证内容：
     * 1. 进度按实际值计算（当前实现不裁剪为100%，保留真实值）
     * 2. 正常更新
     */
    @Test
    @DisplayName("updateProgress - current大于total按实际值计算")
    void testUpdateProgress_CurrentGreaterThanTotal() {
        // Arrange
        when(taskMapper.updateById(any(SysTask.class))).thenReturn(1);

        // Act
        progressCallback.updateProgress(150, 100, "超额完成");

        // Assert - 实际行为：按公式计算，不裁剪
        verify(taskMapper, times(1)).updateById(argThat(task ->
                task.getProgress() == 150 && task.getProcessedFiles() == 150
        ));
    }

    /**
     * 测试进度更新 - 当前进度为负数
     * 测试场景：current参数为负数
     * 验证内容：
     * 1. 进度计算为负数（业务上不应发生）
     * 2. 仍正常更新数据库
     */
    @Test
    @DisplayName("updateProgress - 当前进度为负数")
    void testUpdateProgress_NegativeCurrent() {
        // Arrange
        when(taskMapper.updateById(any(SysTask.class))).thenReturn(1);

        // Act
        progressCallback.updateProgress(-10, 100, "异常");

        // Assert
        verify(taskMapper, times(1)).updateById(argThat(task ->
                task.getProgress() == -10 && task.getProcessedFiles() == -10
        ));
    }

    /**
     * 测试多次连续小进度更新 - 触发节流
     * 测试场景：连续调用多次updateProgress，每次进度变化<5%
     * 验证内容：
     * 1. 数据库updateById只在首次调用时执行
     * 2. 后续调用被节流
     */
    @Test
    @DisplayName("updateProgress - 连续小进度更新触发节流")
    void testUpdateProgress_MultipleSmallUpdates() {
        // Arrange
        when(taskMapper.updateById(any(SysTask.class))).thenReturn(1);

        // Act - 连续更新，每次进度变化<5%
        progressCallback.updateProgress(1, 100, "1%");
        progressCallback.updateProgress(2, 100, "2%");
        progressCallback.updateProgress(3, 100, "3%");
        progressCallback.updateProgress(4, 100, "4%");

        // Assert - 只在第一次更新
        verify(taskMapper, times(1)).updateById(argThat(task ->
                task.getProgress() == 1
        ));
    }

    /**
     * 测试消息参数不影响更新逻辑
     * 测试场景：传入不同的message参数
     * 验证内容：
     * 1. message参数仅用于日志记录
     * 2. 首次调用会触发更新（因为是首次）
     * 3. 第二次调用由于进度变化>=5%，也会更新
     */
    @Test
    @DisplayName("updateProgress - message参数不影响更新逻辑")
    void testUpdateProgress_MessageParameter() {
        // Arrange
        when(taskMapper.updateById(any(SysTask.class))).thenReturn(1);

        // Act
        progressCallback.updateProgress(10, 100, "开始");
        reset(taskMapper);
        progressCallback.updateProgress(15, 100, "处理中");

        // Assert - 进度变化=5% (15-10=5)，满足>=5%条件，会更新
        verify(taskMapper, times(1)).updateById(argThat(task ->
                task.getProgress() == 15
        ));
    }
}
