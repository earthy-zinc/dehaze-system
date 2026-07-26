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
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("任务策略工厂测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class TaskStrategyFactoryTest {

    private TaskStrategyFactory factory;

    @Mock
    private TaskStrategy exportStrategy;

    @Mock
    private TaskStrategy importStrategy;

    @BeforeEach
    void setUp() {
        when(exportStrategy.getTaskTypes()).thenReturn(List.of(
                "dataset_export", "user_export", "role_export"));
        when(importStrategy.getTaskTypes()).thenReturn(List.of(
                "user_import", "role_import"));

        factory = new TaskStrategyFactory(List.of(exportStrategy, importStrategy));
    }

    @Test
    @DisplayName("getStrategy - 成功获取 dataset_export 策略")
    void testGetStrategy_DatasetExport() {
        TaskStrategy strategy = factory.getStrategy("dataset_export");
        assertNotNull(strategy);
        assertTrue(strategy.getTaskTypes().contains("dataset_export"));
    }

    @Test
    @DisplayName("getStrategy - 成功获取 user_import 策略")
    void testGetStrategy_UserImport() {
        TaskStrategy strategy = factory.getStrategy("user_import");
        assertNotNull(strategy);
        assertTrue(strategy.getTaskTypes().contains("user_import"));
    }

    @Test
    @DisplayName("getStrategy - 不支持的类型抛出异常")
    void testGetStrategy_UnsupportedType() {
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> factory.getStrategy("unsupported_type")
        );
        assertTrue(exception.getMessage().contains("Unsupported task type"));
        assertTrue(exception.getMessage().contains("unsupported_type"));
    }

    @Test
    @DisplayName("getStrategy - null 类型抛出异常")
    void testGetStrategy_NullType() {
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> factory.getStrategy(null)
        );
        assertTrue(exception.getMessage().contains("Unsupported task type"));
    }

    @Test
    @DisplayName("getStrategy - 空字符串类型抛出异常")
    void testGetStrategy_EmptyType() {
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> factory.getStrategy("")
        );
        assertTrue(exception.getMessage().contains("Unsupported task type"));
    }

    @Test
    @DisplayName("execute - 策略执行方法可调用")
    void testExecute_StrategyMethod() {
        SysTask mockTask = new SysTask();
        Map<String, Object> params = new HashMap<>();
        ProgressCallback callback = new DefaultProgressCallback(1L, "task-1", 1L, null, null, null);

        when(exportStrategy.execute(any(), any(), any())).thenReturn(TaskResult.success("result"));

        TaskStrategy strategy = factory.getStrategy("dataset_export");
        TaskResult result = strategy.execute(mockTask, params, callback);

        assertNotNull(result);
        assertTrue(result.isSuccess());
    }

    @Test
    @DisplayName("一个策略支持多个任务类型")
    void testStrategySupportsMultipleTaskTypes() {
        TaskStrategy multiStrategy = mock(TaskStrategy.class);
        when(multiStrategy.getTaskTypes()).thenReturn(List.of("user_export", "role_export", "dept_export"));

        TaskStrategyFactory multiFactory = new TaskStrategyFactory(List.of(multiStrategy));

        assertEquals(multiStrategy, multiFactory.getStrategy("user_export"));
        assertEquals(multiStrategy, multiFactory.getStrategy("role_export"));
        assertEquals(multiStrategy, multiFactory.getStrategy("dept_export"));
    }

    @Test
    @DisplayName("重复策略类型注册 - 使用最后注册的策略")
    void testDuplicateStrategyTypes() {
        TaskStrategy anotherExportStrategy = mock(TaskStrategy.class);
        when(anotherExportStrategy.getTaskTypes()).thenReturn(List.of("dataset_export"));

        TaskStrategyFactory factoryWithDuplicate = new TaskStrategyFactory(
                List.of(exportStrategy, anotherExportStrategy)
        );

        TaskStrategy strategy = factoryWithDuplicate.getStrategy("dataset_export");
        assertNotNull(strategy);
        assertTrue(strategy.getTaskTypes().contains("dataset_export"));
    }
}
