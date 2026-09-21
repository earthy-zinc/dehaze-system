package com.pei.dehaze.common.util;

import com.pei.dehaze.model.dto.FileDTO;
import com.pei.dehaze.model.dto.ItemFileDTO;
import com.pei.dehaze.service.ImageProcessingService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mock.web.MockMultipartFile;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@DisplayName("FileDTOFactory 单元测试")
class FileDTOFactoryTest {

    /**
     * 最小合法 JPEG 文件头（FF D8 FF）：图片扩展名走文件头魔数校验，
     * 测试夹具必须携带真实魔数而非任意文本内容
     */
    private static final byte[] JPEG_CONTENT = {
            (byte) 0xFF, (byte) 0xD8, (byte) 0xFF, (byte) 0xE0, 0x00, 0x10
    };

    @Mock
    private ImageProcessingService imageProcessingService;

    private FileDTOFactory fileDTOFactory;

    @TempDir
    Path tempDir;

    @BeforeEach
    void setUp() {
        fileDTOFactory = new FileDTOFactory(imageProcessingService);
    }

    @Test
    @DisplayName("从 MultipartFile 创建 FileDTO - 成功")
    void createFileBO_fromMultipartFile_shouldSucceed() {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "test.jpg",
                "image/jpeg",
                JPEG_CONTENT
        );

        FileDTO result = fileDTOFactory.createFileDTO(file, "upload/20250120");

        assertNotNull(result);
        assertEquals("test.jpg", result.getName());
        assertEquals("jpg", result.getExtension());
        assertNotNull(result.getMd5());
        assertNotNull(result.getObjectName());
        assertTrue(result.getObjectName().startsWith("upload/20250120/"));
    }

    @Test
    @DisplayName("从 File 创建 FileDTO - 成功")
    void createFileBO_fromFile_shouldSucceed() throws IOException {
        File testFile = tempDir.resolve("test.png").toFile();
        try (FileWriter writer = new FileWriter(testFile)) {
            writer.write("test content");
        }

        FileDTO result = fileDTOFactory.createFileDTO(testFile, "upload/20250120");

        assertNotNull(result);
        assertEquals("test.png", result.getName());
        assertEquals("png", result.getExtension());
        assertNotNull(result.getMd5());
        assertTrue(result.getSize() > 0);
    }

    @Test
    @DisplayName("创建 ItemFileDTO - 包含图片信息")
    void createItemFileDTO_shouldIncludeImageInfo() {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "test.jpg",
                "image/jpeg",
                JPEG_CONTENT
        );

        when(imageProcessingService.getImageDimensions(any(File.class)))
                .thenReturn(new int[]{800, 600});

        ItemFileDTO result = fileDTOFactory.createItemFileDTO(
                file, "dataset1", "clear", "description", "outdoor", "light"
        );

        assertNotNull(result);
        assertEquals("test.jpg", result.getName());
        assertEquals("clear", result.getType());
        assertEquals("description", result.getDescription());
        assertEquals("outdoor", result.getSceneType());
        assertEquals("light", result.getHazeLevel());
        assertEquals(800, result.getWidth());
        assertEquals(600, result.getHeight());
    }

    @Test
    @DisplayName("创建 ItemFileDTO - 图片宽高为0时设置为null")
    void createItemFileDTO_whenDimensionsZero_shouldSetNull() {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "test.jpg",
                "image/jpeg",
                JPEG_CONTENT
        );

        when(imageProcessingService.getImageDimensions(any(File.class)))
                .thenReturn(new int[]{0, 0});

        ItemFileDTO result = fileDTOFactory.createItemFileDTO(
                file, "dataset1", "hazy", null, null, null
        );

        assertNull(result.getWidth());
        assertNull(result.getHeight());
    }
}
