package com.pei.dehaze.service.strategy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 任务结果单元测试
 * 测试目的：验证TaskResult的静态工厂方法及属性设置
 * 测试范围：
 * 1. success() 静态方法创建成功结果
 * 2. failure() 静态方法创建失败结果
 * 3. 带 metadata 的成功结果
 *
 * @author earthy-zinc
 * @since 2026-01-20
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("任务结果测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class TaskResultTest {

    // ==================== 成功结果测试 ====================

    /**
     * 测试创建成功结果 - 基本场景
     * 测试场景：使用success(String data)静态方法创建成功结果
     * 验证内容：
     * 1. success标志为true
     * 2. data字段值正确
     * 3. errorMessage为null
     * 4. metadata为null
     */
    @Test
    @DisplayName("success(String) - 创建基本成功结果")
    void testSuccess_Basic() {
        // Arrange
        String resultData = "http://example.com/export.zip";

        // Act
        TaskResult result = TaskResult.success(resultData);

        // Assert
        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertEquals(resultData, result.getData());
        assertNull(result.getErrorMessage());
        assertNull(result.getMetadata());
    }

    /**
     * 测试创建成功结果 - 空数据
     * 测试场景：使用success方法，传入空字符串作为数据
     * 验证内容：
     * 1. success标志为true
     * 2. data字段为空字符串（非null）
     */
    @Test
    @DisplayName("success(String) - 创建空数据成功结果")
    void testSuccess_EmptyData() {
        // Arrange
        String emptyData = "";

        // Act
        TaskResult result = TaskResult.success(emptyData);

        // Assert
        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertEquals(emptyData, result.getData());
        assertNull(result.getErrorMessage());
    }

    /**
     * 测试创建成功结果 - null数据
     * 测试场景：使用success方法，传入null作为数据
     * 验证内容：
     * 1. success标志为true
     * 2. data字段为null
     */
    @Test
    @DisplayName("success(String) - 创建null数据成功结果")
    void testSuccess_NullData() {
        // Act
        TaskResult result = TaskResult.success(null);

        // Assert
        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertNull(result.getData());
    }

    /**
     * 测试创建成功结果 - 带元数据
     * 测试场景：使用success(String data, Map<String, Object> metadata)方法创建成功结果
     * 验证内容：
     * 1. success标志为true
     * 2. data字段值正确
     * 3. metadata字段包含所有键值对
     */
    @Test
    @DisplayName("success(String, Map) - 创建带元数据的成功结果")
    void testSuccess_WithMetadata() {
        // Arrange
        String resultData = "http://example.com/export.zip";
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("fileSize", 1024000L);
        metadata.put("fileName", "export.zip");
        metadata.put("processedCount", 100);

        // Act
        TaskResult result = TaskResult.success(resultData, metadata);

        // Assert
        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertEquals(resultData, result.getData());
        assertNull(result.getErrorMessage());
        assertNotNull(result.getMetadata());
        assertEquals(3, result.getMetadata().size());
        assertEquals(1024000L, result.getMetadata().get("fileSize"));
        assertEquals("export.zip", result.getMetadata().get("fileName"));
        assertEquals(100, result.getMetadata().get("processedCount"));
    }

    /**
     * 测试创建成功结果 - 空元数据Map
     * 测试场景：传入空的Map作为metadata参数
     * 验证内容：
     * 1. success标志为true
     * 2. metadata字段为空Map（非null）
     */
    @Test
    @DisplayName("success(String, Map) - 空元数据Map")
    void testSuccess_EmptyMetadata() {
        // Arrange
        String resultData = "http://example.com/export.zip";
        Map<String, Object> emptyMetadata = new HashMap<>();

        // Act
        TaskResult result = TaskResult.success(resultData, emptyMetadata);

        // Assert
        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertEquals(resultData, result.getData());
        assertNotNull(result.getMetadata());
        assertTrue(result.getMetadata().isEmpty());
    }

    /**
     * 测试创建成功结果 - null元数据Map
     * 测试场景：传入null作为metadata参数
     * 验证内容：
     * 1. success标志为true
     * 2. metadata字段为null
     */
    @Test
    @DisplayName("success(String, Map) - null元数据")
    void testSuccess_NullMetadata() {
        // Arrange
        String resultData = "http://example.com/export.zip";

        // Act
        TaskResult result = TaskResult.success(resultData, null);

        // Assert
        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertEquals(resultData, result.getData());
        assertNull(result.getMetadata());
    }

    /**
     * 测试创建成功结果 - 复杂元数据
     * 测试场景：metadata包含嵌套对象和集合
     * 验证内容：
     * 1. success标志为true
     * 2. metadata字段包含复杂数据结构
     */
    @Test
    @DisplayName("success(String, Map) - 复杂元数据结构")
    void testSuccess_ComplexMetadata() {
        // Arrange
        String resultData = "http://example.com/export.zip";
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("fileSize", 2048000L);
        metadata.put("fileName", "complex-export.zip");

        Map<String, String> fileInfo = new HashMap<>();
        fileInfo.put("format", "zip");
        fileInfo.put("encoding", "UTF-8");
        metadata.put("fileInfo", fileInfo);

        // Act
        TaskResult result = TaskResult.success(resultData, metadata);

        // Assert
        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertNotNull(result.getMetadata());
        assertEquals(3, result.getMetadata().size());
        assertEquals(2048000L, result.getMetadata().get("fileSize"));
        assertNotNull(result.getMetadata().get("fileInfo"));
        @SuppressWarnings("unchecked")
        Map<String, String> retrievedFileInfo = (Map<String, String>) result.getMetadata().get("fileInfo");
        assertEquals("zip", retrievedFileInfo.get("format"));
        assertEquals("UTF-8", retrievedFileInfo.get("encoding"));
    }

    // ==================== 失败结果测试 ====================

    /**
     * 测试创建失败结果 - 基本场景
     * 测试场景：使用failure(String errorMessage)静态方法创建失败结果
     * 验证内容：
     * 1. success标志为false
     * 2. errorMessage字段值正确
     * 3. data为null
     * 4. metadata为null
     */
    @Test
    @DisplayName("failure(String) - 创建基本失败结果")
    void testFailure_Basic() {
        // Arrange
        String errorMsg = "导出失败：文件不存在";

        // Act
        TaskResult result = TaskResult.failure(errorMsg);

        // Assert
        assertNotNull(result);
        assertFalse(result.isSuccess());
        assertEquals(errorMsg, result.getErrorMessage());
        assertNull(result.getData());
        assertNull(result.getMetadata());
    }

    /**
     * 测试创建失败结果 - 空错误消息
     * 测试场景：使用failure方法，传入空字符串作为错误消息
     * 验证内容：
     * 1. success标志为false
     * 2. errorMessage为空字符串（非null）
     */
    @Test
    @DisplayName("failure(String) - 空错误消息")
    void testFailure_EmptyMessage() {
        // Arrange
        String emptyMsg = "";

        // Act
        TaskResult result = TaskResult.failure(emptyMsg);

        // Assert
        assertNotNull(result);
        assertFalse(result.isSuccess());
        assertEquals(emptyMsg, result.getErrorMessage());
        assertNull(result.getData());
    }

    /**
     * 测试创建失败结果 - null错误消息
     * 测试场景：使用failure方法，传入null作为错误消息
     * 验证内容：
     * 1. success标志为false
     * 2. errorMessage为null
     */
    @Test
    @DisplayName("failure(String) - null错误消息")
    void testFailure_NullMessage() {
        // Act
        TaskResult result = TaskResult.failure(null);

        // Assert
        assertNotNull(result);
        assertFalse(result.isSuccess());
        assertNull(result.getErrorMessage());
        assertNull(result.getData());
    }

    /**
     * 测试创建失败结果 - 详细错误消息
     * 测试场景：传入详细的错误消息，包含堆栈信息或具体原因
     * 验证内容：
     * 1. success标志为false
     * 2. errorMessage完整保留
     */
    @Test
    @DisplayName("failure(String) - 详细错误消息")
    void testFailure_DetailedMessage() {
        // Arrange
        String detailedMsg = "导出失败：内存不足\n\tat com.pei.dehaze.export.ExportService.export(ExportService.java:123)\n\tCaused by: java.lang.OutOfMemoryError: Java heap space";

        // Act
        TaskResult result = TaskResult.failure(detailedMsg);

        // Assert
        assertNotNull(result);
        assertFalse(result.isSuccess());
        assertEquals(detailedMsg, result.getErrorMessage());
        assertTrue(result.getErrorMessage().contains("内存不足"));
    }

    // ==================== 边界场景测试 ====================

    /**
     * 测试多个结果对象互不影响
     * 测试场景：创建多个TaskResult对象，修改其中一个对象的属性不影响其他对象
     * 验证内容：
     * 1. 每个TaskResult对象是独立的
     * 2. 对象间不会共享数据
     */
    @Test
    @DisplayName("多个结果对象 - 互不影响")
    void testMultipleResults_Independence() {
        // Arrange & Act
        TaskResult result1 = TaskResult.success("data1");
        TaskResult result2 = TaskResult.success("data2");
        TaskResult result3 = TaskResult.failure("error");

        // Assert
        assertEquals("data1", result1.getData());
        assertEquals("data2", result2.getData());
        assertEquals("error", result3.getErrorMessage());
        assertTrue(result1.isSuccess());
        assertTrue(result2.isSuccess());
        assertFalse(result3.isSuccess());
    }

    /**
     * 测试Builder模式创建结果
     * 测试场景：直接使用TaskResult.builder()创建自定义结果
     * 验证内容：
     * 1. 可以同时设置success和errorMessage
     * 2. 可以同时设置data和metadata
     */
    @Test
    @DisplayName("builder - 自定义结果创建")
    void testBuilder_CustomResult() {
        // Arrange
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("custom", "value");

        // Act
        TaskResult result = TaskResult.builder()
                .success(true)
                .data("custom-data")
                .errorMessage("custom-error")
                .metadata(metadata)
                .build();

        // Assert
        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertEquals("custom-data", result.getData());
        assertEquals("custom-error", result.getErrorMessage());
        assertNotNull(result.getMetadata());
        assertEquals("value", result.getMetadata().get("custom"));
    }
}
