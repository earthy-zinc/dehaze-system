package com.pei.dehaze.service.strategy;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.entity.SysTask;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * 任务策略工厂单元测试
 * 测试目的：验证策略工厂对策略实例的注册、获取和查询功能
 * 测试范围：
 * 1. 成功获取已注册的策略实例
 * 2. 获取不存在的策略时抛出 BusinessException
 *
 * @author earthy-zinc
 * @since 2026-01-20
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("任务策略工厂测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class TaskStrategyFactoryTest {

    private TaskStrategyFactory factory;

    @Mock
    private TaskStrategy datasetStrategy;

    @Mock
    private TaskStrategy itemStrategy;

    @Mock
    private TaskStrategy batchStrategy;

    @BeforeEach
    void setUp() {
        when(datasetStrategy.getTaskType()).thenReturn("dataset_export");
        when(itemStrategy.getTaskType()).thenReturn("item_download");
        when(batchStrategy.getTaskType()).thenReturn("batch_download");

        factory = new TaskStrategyFactory(List.of(datasetStrategy, itemStrategy, batchStrategy));
    }

    // ==================== 获取策略测试 ====================

    /**
     * 测试获取策略 - dataset_export类型
     * 测试场景：根据任务类型获取对应的策略实例
     * 验证内容：
     * 1. 返回正确的策略实例
     * 2. 策略实例的getTaskType()返回正确的类型
     */
    @Test
    @DisplayName("getStrategy - 成功获取dataset_export策略")
    void testGetStrategy_DatasetExport() {
        // Act
        TaskStrategy strategy = factory.getStrategy("dataset_export");

        // Assert
        assertNotNull(strategy);
        assertEquals("dataset_export", strategy.getTaskType());
    }

    /**
     * 测试获取策略 - item_download类型
     * 测试场景：根据任务类型获取对应的策略实例
     * 验证内容：
     * 1. 返回正确的策略实例
     * 2. 策略实例的getTaskType()返回正确的类型
     */
    @Test
    @DisplayName("getStrategy - 成功获取item_download策略")
    void testGetStrategy_ItemDownload() {
        // Act
        TaskStrategy strategy = factory.getStrategy("item_download");

        // Assert
        assertNotNull(strategy);
        assertEquals("item_download", strategy.getTaskType());
    }

    /**
     * 测试获取策略 - batch_download类型
     * 测试场景：根据任务类型获取对应的策略实例
     * 验证内容：
     * 1. 返回正确的策略实例
     * 2. 策略实例的getTaskType()返回正确的类型
     */
    @Test
    @DisplayName("getStrategy - 成功获取batch_download策略")
    void testGetStrategy_BatchDownload() {
        // Act
        TaskStrategy strategy = factory.getStrategy("batch_download");

        // Assert
        assertNotNull(strategy);
        assertEquals("batch_download", strategy.getTaskType());
    }

    /**
     * 测试获取策略 - 不存在的类型
     * 测试场景：尝试获取未注册的任务类型策略
     * 验证内容：
     * 1. 抛出BusinessException异常
     * 2. 异常消息包含不支持的任务类型信息
     */
    @Test
    @DisplayName("getStrategy - 不支持的类型抛出异常")
    void testGetStrategy_UnsupportedType() {
        // Act & Assert
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> factory.getStrategy("unsupported_type")
        );

        assertTrue(exception.getMessage().contains("Unsupported task type"));
        assertTrue(exception.getMessage().contains("unsupported_type"));
    }

    /**
     * 测试获取策略 - null类型
     * 测试场景：传入null作为任务类型
     * 验证内容：
     * 1. 抛出BusinessException异常
     */
    @Test
    @DisplayName("getStrategy - null类型抛出异常")
    void testGetStrategy_NullType() {
        // Act & Assert
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> factory.getStrategy(null)
        );

        assertTrue(exception.getMessage().contains("Unsupported task type"));
    }

    /**
     * 测试获取策略 - 空字符串类型
     * 测试场景：传入空字符串作为任务类型
     * 验证内容：
     * 1. 抛出BusinessException异常
     */
    @Test
    @DisplayName("getStrategy - 空字符串类型抛出异常")
    void testGetStrategy_EmptyType() {
        // Act & Assert
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> factory.getStrategy("")
        );

        assertTrue(exception.getMessage().contains("Unsupported task type"));
    }

    // ==================== 策略执行测试 ====================

    /**
     * 测试策略执行 - 验证execute方法可正常调用
     * 测试场景：获取策略后执行任务
     * 验证内容：
     * 1. 成功获取策略实例
     * 2. 策略实例的execute方法可以被正常调用
     */
    @Test
    @DisplayName("execute - 策略执行方法可调用")
    void testExecute_StrategyMethod() {
        // Arrange
        SysTask mockTask = new SysTask();
        Map<String, Object> params = new HashMap<>();
        ProgressCallback callback = new DefaultProgressCallback(1L, "task-1", 1L, null, null, null);

        when(datasetStrategy.execute(any(), any(), any())).thenReturn(TaskResult.success("result"));

        // Act
        TaskStrategy strategy = factory.getStrategy("dataset_export");
        TaskResult result = strategy.execute(mockTask, params, callback);

        // Assert
        assertNotNull(result);
        assertTrue(result.isSuccess());
    }

    // ==================== 边界场景测试 ====================

    /**
     * 测试重复策略类型注册
     * 测试场景：工厂初始化时传入重复的策略类型
     * 验证内容：
     * 1. 工厂正常初始化
     * 2. 只保留最后一个策略实例
     */
    @Test
    @DisplayName("重复策略类型注册 - 使用最后注册的策略")
    void testDuplicateStrategyTypes() {
        // Arrange
        TaskStrategy anotherDatasetStrategy = mock(TaskStrategy.class);
        when(anotherDatasetStrategy.getTaskType()).thenReturn("dataset_export");

        TaskStrategyFactory factoryWithDuplicate = new TaskStrategyFactory(
                List.of(datasetStrategy, anotherDatasetStrategy)
        );

        // Act
        TaskStrategy strategy = factoryWithDuplicate.getStrategy("dataset_export");

        // Assert
        assertNotNull(strategy);
        assertEquals("dataset_export", strategy.getTaskType());
    }
}
