package com.pei.dehaze.service;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.bo.FileBO;
import com.pei.dehaze.service.impl.file.LocalFileService;
import com.pei.dehaze.service.impl.file.MinioFileService;
import io.minio.MinioClient;
import io.minio.PutObjectArgs;
import io.minio.RemoveObjectArgs;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * 文件存储服务测试
 * 测试目的：验证MinIO和本地文件存储的文件操作
 * 测试场景：
 * 1. 文件上传成功/失败处理
 * 2. 文件删除操作
 * 3. 文件下载操作
 * 4. 异常情况处理
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("文件存储服务测试")
class FileStorageServiceTest {

    /**
     * MinIO文件服务测试
     */
    @Nested
    @DisplayName("MinIO文件服务测试")
    class MinioFileServiceTests {

        @Mock
        private MinioClient minioClient;

        private MinioFileService minioFileService;

        @BeforeEach
        void setUp() throws Exception {
            minioFileService = new MinioFileService();
            // 使用反射设置私有字段
            setPrivateField(minioFileService, "minioClient", minioClient);
            setPrivateField(minioFileService, "bucketName", "test-bucket");
            setPrivateField(minioFileService, "endpoint", "http://localhost:9000");
            setPrivateField(minioFileService, "baseUrl", "http://localhost:9000/test-bucket");
        }

        /**
         * 测试文件上传成功
         * 验证：上传文件后应返回正确的文件路径
         */
        @Test
        @DisplayName("文件上传成功应返回文件路径")
        void uploadFile_Success_ShouldReturnFilePath() throws Exception {
            // Arrange
            String objectName = "test/image.jpg";
            byte[] content = "test content".getBytes();
            InputStream inputStream = new ByteArrayInputStream(content);
            long fileSize = content.length;
            String contentType = "image/jpeg";

            when(minioClient.putObject(any(PutObjectArgs.class))).thenReturn(null);

            // Act
            String result = minioFileService.uploadFile(objectName, inputStream, fileSize, contentType);

            // Assert
            assertThat(result).isNotNull();
            assertThat(result).contains(objectName);
            verify(minioClient, times(1)).putObject(any(PutObjectArgs.class));
        }

        /**
         * 测试文件上传失败
         * 验证：上传失败时应抛出异常
         */
        @Test
        @DisplayName("文件上传失败应抛出异常")
        void uploadFile_Failure_ShouldThrowException() throws Exception {
            // Arrange
            String objectName = "test/image.jpg";
            byte[] content = "test content".getBytes();
            InputStream inputStream = new ByteArrayInputStream(content);
            long fileSize = content.length;
            String contentType = "image/jpeg";

            when(minioClient.putObject(any(PutObjectArgs.class)))
                    .thenThrow(new RuntimeException("Upload failed"));

            // Act & Assert
            assertThatThrownBy(() ->
                    minioFileService.uploadFile(objectName, inputStream, fileSize, contentType))
                    .isInstanceOf(RuntimeException.class);
        }

        /**
         * 测试文件删除成功
         * 验证：删除文件应返回true
         */
        @Test
        @DisplayName("文件删除成功应返回true")
        void deleteFile_Success_ShouldReturnTrue() throws Exception {
            // Arrange
            String objectName = "test/image.jpg";
            doNothing().when(minioClient).removeObject(any(RemoveObjectArgs.class));

            // Act
            boolean result = minioFileService.deleteFile(objectName);

            // Assert
            assertThat(result).isTrue();
            verify(minioClient, times(1)).removeObject(any(RemoveObjectArgs.class));
        }

        /**
         * 测试使用FileBO上传文件
         * 验证：使用FileBO对象上传文件应正确处理
         */
        @Test
        @DisplayName("使用FileBO上传文件应正确处理")
        void uploadFile_WithFileBO_ShouldProcess(@TempDir Path tempDir) throws Exception {
            // Arrange - 创建临时文件
            Path tempFile = tempDir.resolve("test.jpg");
            Files.write(tempFile, "test content".getBytes());

            FileBO fileBO = new FileBO();
            fileBO.setName("test.jpg");
            fileBO.setObjectName("uploads/test.jpg");
            fileBO.setFile(tempFile.toFile());

            when(minioClient.putObject(any(PutObjectArgs.class))).thenReturn(null);

            // Act
            FileBO result = minioFileService.uploadFile(fileBO);

            // Assert
            assertThat(result).isNotNull();
            assertThat(result.getUrl()).isNotNull();
        }
    }

    /**
     * 本地文件服务测试
     */
    @Nested
    @DisplayName("本地文件服务测试")
    class LocalFileServiceTests {

        private LocalFileService localFileService;

        @TempDir
        Path tempDir;

        @BeforeEach
        void setUp() throws Exception {
            localFileService = new LocalFileService();
            // 使用反射设置私有字段
            setPrivateField(localFileService, "uploadPath", tempDir.toString());
            setPrivateField(localFileService, "baseUrl", "http://localhost:8080/files");
        }

        /**
         * 测试本地文件上传成功
         * 验证：上传文件后应在指定目录创建文件
         */
        @Test
        @DisplayName("本地文件上传成功应创建文件")
        void uploadFile_Success_ShouldCreateFile() throws Exception {
            // Arrange
            String objectName = "test/image.jpg";
            byte[] content = "test content".getBytes();
            InputStream inputStream = new ByteArrayInputStream(content);
            long fileSize = content.length;
            String contentType = "image/jpeg";

            // Act
            String result = localFileService.uploadFile(objectName, inputStream, fileSize, contentType);

            // Assert
            assertThat(result).isNotNull();
            assertThat(result).contains("image.jpg");

            // 验证文件已创建
            Path filePath = tempDir.resolve("test").resolve("image.jpg");
            assertThat(Files.exists(filePath)).isTrue();
        }

        /**
         * 测试本地文件删除成功
         * 验证：删除文件后文件应不存在
         */
        @Test
        @DisplayName("本地文件删除成功应移除文件")
        void deleteFile_Success_ShouldRemoveFile() throws Exception {
            // Arrange - 先创建文件
            Path testFile = tempDir.resolve("test.jpg");
            Files.write(testFile, "test content".getBytes());
            assertThat(Files.exists(testFile)).isTrue();

            // Act
            boolean result = localFileService.deleteFile("test.jpg");

            // Assert
            assertThat(result).isTrue();
            assertThat(Files.exists(testFile)).isFalse();
        }

        /**
         * 测试删除不存在的文件
         * 验证：删除不存在的文件应返回true（幂等性设计）
         */
        @Test
        @DisplayName("删除不存在的文件应返回true（幂等性）")
        void deleteFile_NotExists_ShouldReturnTrue() {
            // Arrange
            String nonExistentFile = "non_existent.jpg";

            // Act
            boolean result = localFileService.deleteFile(nonExistentFile);

            // Assert - 根据实际实现，文件不存在视为删除成功（幂等性）
            assertThat(result).isTrue();
        }

        /**
         * 测试本地文件下载
         * 验证：下载文件应返回正确的输入流
         */
        @Test
        @DisplayName("本地文件下载应返回输入流")
        void downloadFile_Success_ShouldReturnInputStream() throws Exception {
            // Arrange - 先创建文件
            Path testFile = tempDir.resolve("download_test.jpg");
            byte[] content = "test content for download".getBytes();
            Files.write(testFile, content);

            // Act & Assert - 使用 try-with-resources 确保流关闭，避免 Windows 临时目录清理失败
            try (InputStream result = localFileService.downLoadFile("download_test.jpg")) {
                assertThat(result).isNotNull();
                byte[] downloadedContent = result.readAllBytes();
                assertThat(downloadedContent).isEqualTo(content);
            }
        }

        /**
         * 测试下载不存在的文件
         * 验证：下载不存在的文件应抛出异常
         */
        @Test
        @DisplayName("下载不存在的文件应抛出异常")
        void downloadFile_NotExists_ShouldThrowException() {
            // Arrange
            String nonExistentFile = "non_existent_download.jpg";

            // Act & Assert
            assertThatThrownBy(() -> localFileService.downLoadFile(nonExistentFile))
                    .isInstanceOf(BusinessException.class)
                    .hasMessageContaining("文件不存在");
        }

        /**
         * 测试上传到嵌套目录
         * 验证：上传到嵌套目录应自动创建目录结构
         */
        @Test
        @DisplayName("上传到嵌套目录应自动创建目录结构")
        void uploadFile_NestedDirectory_ShouldCreateDirectories() throws Exception {
            // Arrange
            String objectName = "level1/level2/level3/nested.jpg";
            byte[] content = "nested content".getBytes();
            InputStream inputStream = new ByteArrayInputStream(content);
            long fileSize = content.length;
            String contentType = "image/jpeg";

            // Act
            String result = localFileService.uploadFile(objectName, inputStream, fileSize, contentType);

            // Assert
            assertThat(result).isNotNull();
            Path filePath = tempDir.resolve("level1/level2/level3/nested.jpg");
            assertThat(Files.exists(filePath)).isTrue();
        }
    }

    /**
     * 边界条件测试
     */
    @Nested
    @DisplayName("边界条件测试")
    class EdgeCaseTests {

        private LocalFileService localFileService;

        @TempDir
        Path tempDir;

        @BeforeEach
        void setUp() throws Exception {
            localFileService = new LocalFileService();
            setPrivateField(localFileService, "uploadPath", tempDir.toString());
            setPrivateField(localFileService, "baseUrl", "http://localhost:8080/files");
        }

        /**
         * 测试上传空文件
         * 验证：上传空文件应正常处理
         */
        @Test
        @DisplayName("上传空文件应正常处理")
        void uploadFile_EmptyContent_ShouldHandle() throws Exception {
            // Arrange
            String objectName = "empty.txt";
            byte[] content = new byte[0];
            InputStream inputStream = new ByteArrayInputStream(content);
            long fileSize = 0;
            String contentType = "text/plain";

            // Act
            String result = localFileService.uploadFile(objectName, inputStream, fileSize, contentType);

            // Assert
            assertThat(result).isNotNull();
            Path filePath = tempDir.resolve("empty.txt");
            assertThat(Files.exists(filePath)).isTrue();
            assertThat(Files.size(filePath)).isZero();
        }

        /**
         * 测试特殊字符文件名
         * 验证：包含特殊字符的文件名应正确处理
         */
        @Test
        @DisplayName("特殊字符文件名应正确处理")
        void uploadFile_SpecialCharacters_ShouldHandle() throws Exception {
            // Arrange - 使用中文文件名
            String objectName = "测试文件_2024.jpg";
            byte[] content = "content".getBytes();
            InputStream inputStream = new ByteArrayInputStream(content);
            long fileSize = content.length;
            String contentType = "image/jpeg";

            // Act
            String result = localFileService.uploadFile(objectName, inputStream, fileSize, contentType);

            // Assert
            assertThat(result).isNotNull();
        }

        /**
         * 测试大文件上传
         * 验证：大文件应能正常上传
         */
        @Test
        @DisplayName("大文件上传应正常处理")
        void uploadFile_LargeFile_ShouldHandle() throws Exception {
            // Arrange - 创建1MB的测试数据
            int size = 1024 * 1024;
            byte[] content = new byte[size];
            for (int i = 0; i < size; i++) {
                content[i] = (byte) (i % 256);
            }
            String objectName = "large_file.bin";
            InputStream inputStream = new ByteArrayInputStream(content);
            String contentType = "application/octet-stream";

            // Act
            String result = localFileService.uploadFile(objectName, inputStream, size, contentType);

            // Assert
            assertThat(result).isNotNull();
            Path filePath = tempDir.resolve("large_file.bin");
            assertThat(Files.exists(filePath)).isTrue();
            assertThat(Files.size(filePath)).isEqualTo(size);
        }
    }

    /**
     * 反射工具方法：设置私有字段
     */
    private static void setPrivateField(Object object, String fieldName, Object value)
            throws NoSuchFieldException, IllegalAccessException {
        Field field = object.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(object, value);
    }
}
