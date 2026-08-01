package com.pei.dehaze.common.util;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;

import static org.junit.jupiter.api.Assertions.*;

/**
 * FilePathBuilder 单元测试
 */
@DisplayName("FilePathBuilder 单元测试")
class FilePathBuilderTest {

    private FilePathBuilder filePathBuilder;

    @BeforeEach
    void setUp() {
        filePathBuilder = new FilePathBuilder();
    }

    @Test
    @DisplayName("构建当日上传路径")
    void buildUploadPath_shouldReturnTodayPath() {
        String path = filePathBuilder.buildUploadPath();

        assertNotNull(path);
        assertTrue(path.startsWith("upload/"));
        assertEquals(15, path.length()); // upload/yyyyMMdd = 15 chars
    }

    @Test
    @DisplayName("构建指定日期上传路径")
    void buildUploadPath_withDateTime_shouldReturnFormattedPath() {
        LocalDateTime dateTime = LocalDateTime.of(2025, 1, 19, 10, 30);

        String path = filePathBuilder.buildUploadPath(dateTime);

        assertEquals("upload/20250119", path);
    }

    @Test
    @DisplayName("构建对象名")
    void buildObjectName_shouldReturnCorrectFormat() {
        String objectName = filePathBuilder.buildObjectName("20250119", "abc123def456", "jpg");

        assertEquals("upload/20250119/abc123def456.jpg", objectName);
    }

    @Test
    @DisplayName("构建缩略图路径")
    void buildThumbnailPath_shouldReturnCorrectFormat() {
        String thumbnailPath = filePathBuilder.buildThumbnailPath("upload/20250119", "thumb123", "jpg");

        assertEquals("thumbnail/upload/20250119/thumb123.jpg", thumbnailPath);
    }

    @Test
    @DisplayName("构建缩略图对象名（从原图对象名）")
    void buildThumbnailObjectName_shouldExtractPathCorrectly() {
        String originObjectName = "upload/20250119/original123.jpg";
        String thumbnailMd5 = "thumb456";

        String thumbnailObjectName = filePathBuilder.buildThumbnailObjectName(originObjectName, thumbnailMd5, "jpg");

        assertEquals("thumbnail/upload/20250119/thumb456.jpg", thumbnailObjectName);
    }

    @Test
    @DisplayName("构建导出路径")
    void buildExportPath_shouldReturnCorrectFormat() {
        String exportPath = filePathBuilder.buildExportPath("task-001");

        assertEquals("exports/task-001.zip", exportPath);
    }
}
