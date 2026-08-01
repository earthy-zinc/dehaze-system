package com.pei.dehaze.service;

import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.vo.ImageUrlVO;
import com.pei.dehaze.service.impl.SysItemFileServiceImpl;
import com.pei.dehaze.service.impl.file.StorageServiceFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.when;

/**
 * 图片文件服务单元测试
 * 测试目的：验证图片文件业务逻辑的正确性
 * 测试范围：
 * 1. 图片转换逻辑（不依赖Mapper）
 * <p>
 * 注意：需要数据库交互的方法（查询、更新、删除）建议使用集成测试
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("图片文件服务测试")
class SysItemFileServiceTest {

    @Mock
    private SysFileService sysFileService;

    @Mock
    private StorageServiceFactory storageServiceFactory;

    @Mock
    private FileService fileService;

    @InjectMocks
    private SysItemFileServiceImpl sysItemFileServiceImpl;

    private SysItemFile mockItemFile;
    private SysFile mockFile;
    private SysFile mockThumbnailFile;

    @BeforeEach
    void setUp() {
        // 初始化Mock数据（object_name + storage，不落库 url）
        mockFile = SysFile.builder()
                .id(1L)
                .name("clear_001.jpg")
                .type("jpg")
                .objectName("images/clear_001.jpg")
                .storage("minio")
                .size("2560000")
                .build();

        mockThumbnailFile = SysFile.builder()
                .id(2L)
                .name("clear_001_thumb.jpg")
                .type("jpg")
                .objectName("thumbs/clear_001_thumb.jpg")
                .storage("minio")
                .size("50000")
                .build();

        mockItemFile = new SysItemFile();
        mockItemFile.setId(1L);
        mockItemFile.setItemId(10L);
        mockItemFile.setFileId(1L);
        mockItemFile.setThumbnailFileId(2L);
        mockItemFile.setType("clear");
        mockItemFile.setDescription("清晰的城市街道图片");
        mockItemFile.setSceneType("outdoor");
        mockItemFile.setHazeLevel("none");
        mockItemFile.setWidth(1920);
        mockItemFile.setHeight(1080);
        mockItemFile.setUsageCount(5L);
        mockItemFile.setCreateTime(LocalDateTime.now());

        // StorageServiceFactory 按 storage 返回 mock FileService，getUrl 动态拼接
        // 使用 lenient 避免不调用这些 stub 的测试方法触发 UnnecessaryStubbing
        lenient().when(storageServiceFactory.get(anyString())).thenReturn(fileService);
        lenient().when(fileService.getUrl("images/clear_001.jpg")).thenReturn("http://cdn.example.com/images/clear_001.jpg");
        lenient().when(fileService.getUrl("thumbs/clear_001_thumb.jpg")).thenReturn("http://cdn.example.com/thumbs/clear_001_thumb.jpg");
    }

    /**
     * 测试将SysItemFile转换为ImageUrlVO
     * 测试场景：将实体对象转换为VO对象
     * 验证内容：
     * 1. 所有字段正确转换
     * 2. 文件信息正确填充（url 由 storageService.getUrl(objectName) 动态拼接）
     */
    @Test
    @DisplayName("将SysItemFile转换为ImageUrlVO")
    void testConvertToImageUrlVO() {
        // Arrange
        Map<Long, SysFile> fileMap = Map.of(1L, mockFile, 2L, mockThumbnailFile);

        // Act
        ImageUrlVO result = sysItemFileServiceImpl.convertToImageUrlVO(mockItemFile, fileMap);

        // Assert
        assertNotNull(result);
        assertEquals(1L, result.getId());
        assertEquals(10L, result.getItemId());
        assertEquals("clear", result.getType());
        assertEquals("清晰的城市街道图片", result.getDescription());
        assertEquals("outdoor", result.getSceneType());
        assertEquals("none", result.getHazeLevel());
        assertEquals(1920, result.getWidth());
        assertEquals(1080, result.getHeight());
        assertEquals(5L, result.getUsageCount());
        assertEquals("clear_001.jpg", result.getFileName());
        assertEquals("jpg", result.getFormat());
        assertEquals("http://cdn.example.com/images/clear_001.jpg", result.getUrl());
        assertEquals("http://cdn.example.com/thumbs/clear_001_thumb.jpg", result.getThumbnailUrl());
    }

    /**
     * 测试转换无缩略图的图片
     * 测试场景：图片没有缩略图（thumbnailFileId为null）
     * 验证内容：
     * 1. 正常转换
     * 2. thumbnailUrl为null
     */
    @Test
    @DisplayName("将SysItemFile转换为ImageUrlVO - 无缩略图")
    void testConvertToImageUrlVO_NoThumbnail() {
        // Arrange
        mockItemFile.setThumbnailFileId(null);
        Map<Long, SysFile> fileMap = Map.of(1L, mockFile);

        // Act
        ImageUrlVO result = sysItemFileServiceImpl.convertToImageUrlVO(mockItemFile, fileMap);

        // Assert
        assertNotNull(result);
        assertEquals(1L, result.getId());
        assertNull(result.getThumbnailUrl());
    }

    /**
     * 测试转换无文件信息的图片
     * 测试场景：文件信息不存在
     * 验证内容：
     * 1. 正常转换
     * 2. 文件相关字段为null或默认值
     */
    @Test
    @DisplayName("将SysItemFile转换为ImageUrlVO - 无文件信息")
    void testConvertToImageUrlVO_NoFile() {
        // Arrange: 文件Map中不包含对应的fileId，模拟文件信息不存在
        Map<Long, SysFile> fileMap = Map.of();

        // Act
        ImageUrlVO result = sysItemFileServiceImpl.convertToImageUrlVO(mockItemFile, fileMap);

        // Assert
        assertNotNull(result);
        assertEquals(1L, result.getId());
        assertEquals(10L, result.getItemId());
        assertEquals("clear", result.getType());
        assertNull(result.getFileName());
        assertNull(result.getFormattedSize());
        assertNull(result.getFormat());
        assertNull(result.getUrl());
    }

    /**
     * 测试转换使用次数为null的图片
     * 测试场景：usageCount字段为null
     * 验证内容：
     * 1. 正常转换
     * 2. usageCount默认为0
     */
    @Test
    @DisplayName("将SysItemFile转换为ImageUrlVO - usageCount为null")
    void testConvertToImageUrlVO_NullUsageCount() {
        // Arrange
        mockItemFile.setUsageCount(null);
        Map<Long, SysFile> fileMap = Map.of(1L, mockFile, 2L, mockThumbnailFile);

        // Act
        ImageUrlVO result = sysItemFileServiceImpl.convertToImageUrlVO(mockItemFile, fileMap);

        // Assert
        assertNotNull(result);
        assertEquals(1L, result.getId());
        assertEquals(0L, result.getUsageCount());
    }

    /**
     * 测试转换有雾图片
     * 测试场景：转换有雾类型的图片
     * 验证内容：
     * 1. 类型字段正确
     * 2. 雾霾程度字段正确
     */
    @Test
    @DisplayName("将SysItemFile转换为ImageUrlVO - 有雾图片")
    void testConvertToImageUrlVO_HazyImage() {
        // Arrange
        mockItemFile.setType("hazy");
        mockItemFile.setHazeLevel("moderate");
        Map<Long, SysFile> fileMap = Map.of(1L, mockFile, 2L, mockThumbnailFile);

        // Act
        ImageUrlVO result = sysItemFileServiceImpl.convertToImageUrlVO(mockItemFile, fileMap);

        // Assert
        assertNotNull(result);
        assertEquals(1L, result.getId());
        assertEquals("hazy", result.getType());
        assertEquals("moderate", result.getHazeLevel());
    }
}
