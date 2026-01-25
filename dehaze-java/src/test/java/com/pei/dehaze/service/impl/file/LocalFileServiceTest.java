package com.pei.dehaze.service.impl.file;

import com.pei.dehaze.common.exception.BusinessException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.InjectMocks;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * LocalFileService 单元测试
 * <p>
 * 测试目的：验证 LocalFileService 的文件操作逻辑
 * 测试策略：使用实际文件系统操作，确保功能正常
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("LocalFileService 单元测试")
class LocalFileServiceTest {

    @InjectMocks
    private LocalFileService localFileService;

    @TempDir
    Path tempDir;

    private static final String BASE_URL = "http://localhost:8080/files";
    private static final String UPLOAD_PATH = "/data/files";

    @BeforeEach
    void setUp() {
        // 使用临时目录作为上传路径
        localFileService.setUploadPath(tempDir.toString());
        localFileService.setBaseUrl(BASE_URL);
    }

    // ==================== deleteFile 测试 ====================

    /**
     * 测试删除文件 - 成功删除存在的文件
     * 测试目的：验证能够成功删除存在的文件
     * 测试场景：文件存在且路径合法
     * 验证内容：返回true，文件被删除
     */
    @Test
    @DisplayName("deleteFile - 成功删除存在的文件")
    void testDeleteFile_Success() throws IOException {
        // Given
        String objectName = "test/delete.txt";
        Path filePath = tempDir.resolve(objectName);

        Files.createDirectories(filePath.getParent());
        Files.writeString(filePath, "test content");

        assertThat(Files.exists(filePath)).isTrue();

        // When
        boolean result = localFileService.deleteFile(objectName);

        // Then
        assertThat(result).isTrue();
        assertThat(Files.exists(filePath)).isFalse();
    }

    /**
     * 测试删除文件 - 文件不存在返回true（幂等性）
     * 测试目的：验证当文件不存在时的幂等性处理
     * 测试场景：文件不存在
     * 验证内容：返回true（不抛出异常）
     */
    @Test
    @DisplayName("deleteFile - 文件不存在返回true（幂等性）")
    void testDeleteFile_FileNotExists() {
        // Given
        String nonExistentFile = "test/nonexistent.txt";

        // When
        boolean result = localFileService.deleteFile(nonExistentFile);

        // Then
        assertThat(result).isTrue();
    }

    /**
     * 测试删除文件 - 路径遍历攻击防护
     * 测试目的：验证路径遍历攻击的防护机制
     * 测试场景：尝试使用../跳出uploadPath
     * 验证内容：抛出IllegalArgumentException异常
     */
    @Test
    @DisplayName("deleteFile - 路径遍历攻击防护")
    void testDeleteFile_PathTraversalAttack() {
        // Given
        String maliciousPath = "../../../etc/passwd";

        // When & Then
        assertThatThrownBy(() -> localFileService.deleteFile(maliciousPath))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("无效的文件路径");
    }

    /**
     * 测试删除文件 - 规范化路径的遍历攻击防护
     * 测试目的：验证规范化路径后的路径遍历攻击防护
     * 测试场景：使用././../等跳转字符
     * 验证内容：抛出IllegalArgumentException异常
     */
    @Test
    @DisplayName("deleteFile - 规范化路径的遍历攻击防护")
    void testDeleteFile_NormalizedPathTraversal() {
        // Given
        String maliciousPath = "test/./../../etc/passwd";

        // When & Then
        assertThatThrownBy(() -> localFileService.deleteFile(maliciousPath))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("无效的文件路径");
    }

    /**
     * 测试删除文件 - 删除空目录中的文件
     * 测试目的：验证能够删除嵌套目录中的文件
     * 测试场景：文件位于深层嵌套目录中
     * 验证内容：返回true，文件被删除
     */
    @Test
    @DisplayName("deleteFile - 删除嵌套目录中的文件")
    void testDeleteFile_NestedDirectory() throws IOException {
        // Given
        String objectName = "level1/level2/level3/file.txt";
        Path filePath = tempDir.resolve(objectName);

        Files.createDirectories(filePath.getParent());
        Files.writeString(filePath, "nested content");

        assertThat(Files.exists(filePath)).isTrue();

        // When
        boolean result = localFileService.deleteFile(objectName);

        // Then
        assertThat(result).isTrue();
        assertThat(Files.exists(filePath)).isFalse();
    }

    /**
     * 测试删除文件 - 删除根目录中的文件
     * 测试目的：验证能够删除uploadPath根目录中的文件
     * 测试场景：文件位于uploadPath根目录
     * 验证内容：返回true，文件被删除
     */
    @Test
    @DisplayName("deleteFile - 删除根目录中的文件")
    void testDeleteFile_RootDirectory() throws IOException {
        // Given
        String objectName = "root.txt";
        Path filePath = tempDir.resolve(objectName);

        Files.writeString(filePath, "root file content");

        assertThat(Files.exists(filePath)).isTrue();

        // When
        boolean result = localFileService.deleteFile(objectName);

        // Then
        assertThat(result).isTrue();
        assertThat(Files.exists(filePath)).isFalse();
    }

    /**
     * 测试删除文件 - 文件路径包含特殊字符
     * 测试目的：验证文件名包含特殊字符时的处理
     * 测试场景：文件名包含空格、点等合法字符
     * 验证内容：能够正确删除
     */
    @Test
    @DisplayName("deleteFile - 文件路径包含特殊字符")
    void testDeleteFile_SpecialCharacters() throws IOException {
        // Given
        String objectName = "test/file name with spaces (1).txt";
        Path filePath = tempDir.resolve(objectName);

        Files.createDirectories(filePath.getParent());
        Files.writeString(filePath, "special chars content");

        assertThat(Files.exists(filePath)).isTrue();

        // When
        boolean result = localFileService.deleteFile(objectName);

        // Then
        assertThat(result).isTrue();
        assertThat(Files.exists(filePath)).isFalse();
    }

    /**
     * 测试删除文件 - 重复删除同一文件（幂等性）
     * 测试目的：验证删除操作的幂等性
     * 测试场景：对同一文件调用两次deleteFile
     * 验证内容：两次都返回true
     */
    @Test
    @DisplayName("deleteFile - 重复删除同一文件（幂等性）")
    void testDeleteFile_Idempotent() throws IOException {
        // Given
        String objectName = "test/idempotent.txt";
        Path filePath = tempDir.resolve(objectName);

        Files.createDirectories(filePath.getParent());
        Files.writeString(filePath, "idempotent test");

        assertThat(Files.exists(filePath)).isTrue();

        // When - 第一次删除
        boolean result1 = localFileService.deleteFile(objectName);
        assertThat(result1).isTrue();

        // When - 第二次删除
        boolean result2 = localFileService.deleteFile(objectName);

        // Then
        assertThat(result2).isTrue();
        assertThat(Files.exists(filePath)).isFalse();
    }

    // ==================== downLoadFile 测试 ====================

    /**
     * 测试下载文件 - 成功
     * 测试目的：验证能够成功下载存在的文件
     * 测试场景：文件存在且路径合法
     * 验证内容：返回有效的InputStream
     */
    @Test
    @DisplayName("downLoadFile - 成功")
    void testDownLoadFile_Success() throws IOException {
        // Given
        String objectName = "test/download.txt";
        Path filePath = tempDir.resolve(objectName);

        Files.createDirectories(filePath.getParent());
        Files.writeString(filePath, "download content");

        assertThat(Files.exists(filePath)).isTrue();

        // When
        InputStream result = localFileService.downLoadFile(objectName);

        // Then
        assertThat(result).isNotNull();

        byte[] content = result.readAllBytes();
        assertThat(new String(content)).isEqualTo("download content");

        result.close();
    }

    /**
     * 测试下载文件 - 文件不存在抛出异常
     * 测试目的：验证文件不存在时的异常处理
     * 测试场景：文件不存在
     * 验证内容：抛出BusinessException
     */
    @Test
    @DisplayName("downLoadFile - 文件不存在抛出异常")
    void testDownLoadFile_FileNotExists() {
        // Given
        String nonExistentFile = "test/nonexistent.txt";

        // When & Then
        assertThatThrownBy(() -> localFileService.downLoadFile(nonExistentFile))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("文件不存在");
    }

    /**
     * 测试下载文件 - 路径遍历攻击防护
     * 测试目的：验证下载时的路径遍历攻击防护
     * 测试场景：尝试使用../跳出uploadPath
     * 验证内容：抛出BusinessException或IllegalArgumentException
     */
    @Test
    @DisplayName("downLoadFile - 路径遍历攻击防护")
    void testDownLoadFile_PathTraversalAttack() {
        // Given
        String maliciousPath = "../../../etc/passwd";

        // When & Then
        Throwable thrown = catchThrowable(() -> localFileService.downLoadFile(maliciousPath));
        assertThat(thrown)
                .isNotNull()
                .satisfies(e -> {
                    assertThat(e).isInstanceOfAny(BusinessException.class, IllegalArgumentException.class);
                    if (e instanceof BusinessException) {
                        assertThat(e.getMessage()).contains("文件不存在");
                    } else {
                        assertThat(e.getMessage()).contains("无效的文件路径");
                    }
                });
    }

    /**
     * 测试下载文件 - 文件名包含非法字符抛出异常
     * 测试目的：验证文件名注入攻击防护
     * 测试场景：文件名包含非法字符
     * 验证内容：抛出BusinessException
     */
    @Test
    @DisplayName("downLoadFile - 文件名包含非法字符抛出异常")
    void testDownLoadFile_InvalidFileName() throws IOException {
        // Given
        String objectName = "test/invalid<>file.txt";
        Path filePath = tempDir.resolve(objectName);

        Files.createDirectories(filePath.getParent());
        Files.writeString(filePath, "invalid content");

        assertThat(Files.exists(filePath)).isTrue();

        // When & Then
        assertThatThrownBy(() -> localFileService.downLoadFile(objectName))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("不支持的文件名");
    }

    // ==================== uploadFile 测试 ====================

    /**
     * 测试上传文件 - 成功
     * 测试目的：验证能够成功上传文件
     * 测试场景：创建目录并保存文件
     * 验证内容：返回正确的URL，文件被保存
     */
    @Test
    @DisplayName("uploadFile - 成功")
    void testUploadFile_Success() throws IOException {
        // Given
        String objectName = "test/upload.txt";
        String content = "upload content";
        InputStream inputStream = new ByteArrayInputStream(content.getBytes());
        long fileSize = content.length();
        String contentType = "text/plain";

        // When
        String result = localFileService.uploadFile(objectName, inputStream, fileSize, contentType);

        // Then
        assertThat(result).isNotNull();
        assertThat(result).isEqualTo(BASE_URL + "/" + objectName);

        Path filePath = tempDir.resolve(objectName);
        assertThat(Files.exists(filePath)).isTrue();
        assertThat(Files.readString(filePath)).isEqualTo(content);

        inputStream.close();
    }

    /**
     * 测试上传文件 - 自动创建目录
     * 测试目的：验证上传时自动创建嵌套目录
     * 测试场景：上传到不存在的多级目录
     * 验证内容：目录被创建，文件被保存
     */
    @Test
    @DisplayName("uploadFile - 自动创建目录")
    void testUploadFile_AutoCreateDirectory() throws IOException {
        // Given
        String objectName = "new/deep/dir/file.txt";
        String content = "nested upload";
        InputStream inputStream = new ByteArrayInputStream(content.getBytes());

        // When
        String result = localFileService.uploadFile(objectName, inputStream, content.length(), "text/plain");

        // Then
        assertThat(result).isNotNull();

        Path filePath = tempDir.resolve(objectName);
        assertThat(Files.exists(filePath)).isTrue();
        assertThat(Files.readString(filePath)).isEqualTo(content);

        inputStream.close();
    }
}
