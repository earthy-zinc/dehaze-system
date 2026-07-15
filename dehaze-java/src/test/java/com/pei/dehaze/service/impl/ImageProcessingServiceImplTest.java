package com.pei.dehaze.service.impl;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.service.ImageProcessingService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * ImageProcessingService 单元测试
 */
@DisplayName("ImageProcessingService 单元测试")
class ImageProcessingServiceImplTest {

    private ImageProcessingService imageProcessingService;

    @TempDir
    Path tempDir;

    @BeforeEach
    void setUp() {
        imageProcessingService = new ImageProcessingServiceImpl();
        // 设置默认配置
        ReflectionTestUtils.setField(imageProcessingService, "maxFileSize", 10 * 1024 * 1024L);
        ReflectionTestUtils.setField(imageProcessingService, "thumbnailQuality", 0.5f);
    }

    @Test
    @DisplayName("校验空文件应抛出异常")
    void validateImageFile_withNullFile_shouldThrowException() {
        File nullFile = null;

        BusinessException exception = assertThrows(BusinessException.class,
                () -> imageProcessingService.validateImageFile(nullFile));

        assertEquals("文件不能为空", exception.getMessage());
    }

    @Test
    @DisplayName("校验不存在的文件应抛出异常")
    void validateImageFile_withNonExistentFile_shouldThrowException() {
        File nonExistentFile = new File("/non/existent/path/file.jpg");

        BusinessException exception = assertThrows(BusinessException.class,
                () -> imageProcessingService.validateImageFile(nonExistentFile));

        assertEquals("文件不能为空", exception.getMessage());
    }

    @Test
    @DisplayName("校验超大文件应抛出异常")
    void validateImageFile_withOversizedFile_shouldThrowException() throws IOException {
        // 创建一个超过限制的临时文件
        Path largeFilePath = tempDir.resolve("large.jpg");
        byte[] largeContent = new byte[11 * 1024 * 1024]; // 11MB
        Files.write(largeFilePath, largeContent);

        BusinessException exception = assertThrows(BusinessException.class,
                () -> imageProcessingService.validateImageFile(largeFilePath.toFile()));

        assertTrue(exception.getMessage().contains("文件大小不能超过"));
    }

    @Test
    @DisplayName("校验不支持的格式应抛出异常")
    void validateImageFile_withUnsupportedFormat_shouldThrowException() throws IOException {
        Path unsupportedFile = tempDir.resolve("document.pdf");
        Files.write(unsupportedFile, "PDF content".getBytes());

        BusinessException exception = assertThrows(BusinessException.class,
                () -> imageProcessingService.validateImageFile(unsupportedFile.toFile()));

        assertTrue(exception.getMessage().contains("仅支持"));
    }

    @Test
    @DisplayName("校验有效图片文件应通过")
    void validateImageFile_withValidFile_shouldPass() throws IOException {
        Path validFile = tempDir.resolve("valid.jpg");
        Files.write(validFile, "fake image content".getBytes());

        assertDoesNotThrow(() -> imageProcessingService.validateImageFile(validFile.toFile()));
    }

    @Test
    @DisplayName("校验空MultipartFile应抛出异常")
    void validateImageFile_withEmptyMultipartFile_shouldThrowException() {
        MockMultipartFile emptyFile = new MockMultipartFile(
                "file", "empty.jpg", "image/jpeg", new byte[0]);

        BusinessException exception = assertThrows(BusinessException.class,
                () -> imageProcessingService.validateImageFile(emptyFile));

        assertEquals("文件不能为空", exception.getMessage());
    }

    @Test
    @DisplayName("判断支持的图片格式")
    void isSupportedImageFormat_shouldReturnCorrectResult() {
        assertTrue(imageProcessingService.isSupportedImageFormat("jpg"));
        assertTrue(imageProcessingService.isSupportedImageFormat("jpeg"));
        assertTrue(imageProcessingService.isSupportedImageFormat("png"));
        assertTrue(imageProcessingService.isSupportedImageFormat("gif"));
        assertTrue(imageProcessingService.isSupportedImageFormat("JPG")); // 大小写不敏感
        assertFalse(imageProcessingService.isSupportedImageFormat("pdf"));
        assertFalse(imageProcessingService.isSupportedImageFormat("txt"));
        assertFalse(imageProcessingService.isSupportedImageFormat(null));
    }

    @Test
    @DisplayName("判断文件名是否为图片")
    void isImage_shouldReturnCorrectResult() {
        assertTrue(imageProcessingService.isImage("photo.jpg"));
        assertTrue(imageProcessingService.isImage("image.PNG"));
        assertTrue(imageProcessingService.isImage("animation.gif"));
        assertFalse(imageProcessingService.isImage("document.pdf"));
        assertFalse(imageProcessingService.isImage("noextension"));
        assertFalse(imageProcessingService.isImage(null));
    }

    @Test
    @DisplayName("获取支持的格式列表")
    void getSupportedFormats_shouldReturnNonEmptySet() {
        Set<String> formats = imageProcessingService.getSupportedFormats();

        assertNotNull(formats);
        assertFalse(formats.isEmpty());
        assertTrue(formats.contains("jpg"));
        assertTrue(formats.contains("png"));
    }

    @Test
    @DisplayName("获取最大文件大小")
    void getMaxFileSize_shouldReturnConfiguredValue() {
        long maxSize = imageProcessingService.getMaxFileSize();

        assertEquals(10 * 1024 * 1024L, maxSize);
    }

    @Test
    @DisplayName("解析图片宽高 - 无效文件抛出异常")
    void getImageDimensions_withInvalidFile_shouldThrowException() throws IOException {
        Path textFile = tempDir.resolve("text.jpg");
        Files.write(textFile, "not an image".getBytes());

        assertThrows(BusinessException.class,
                () -> imageProcessingService.getImageDimensions(textFile.toFile()));
    }
}
