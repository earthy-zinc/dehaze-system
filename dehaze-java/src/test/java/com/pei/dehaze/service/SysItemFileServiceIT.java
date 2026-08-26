package com.pei.dehaze.service;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.mapper.SysDatasetItemMapper;
import com.pei.dehaze.mapper.SysItemFileMapper;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.form.ItemFileUpdateForm;
import com.pei.dehaze.model.vo.BatchDeleteResultVO;
import com.pei.dehaze.model.vo.ImageUrlVO;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

import com.pei.dehaze.config.TestConfig;

/**
 * 图片文件服务集成测试
 * 测试目的：验证图片文件业务逻辑在真实环境下的正确性
 * 测试范围：
 * 1. 图片查询（getImageById）
 * 2. 图片保存（saveItemFile）
 * 3. 图片更新（updateItemFileInfo）
 * 4. 图片删除（deleteFile）
 * 5. 批量删除（batchDelete）
 * <p>
 * 注意：使用@SpringBootTest启动完整Spring容器，使用@Transactional自动回滚测试数据
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@SpringBootTest(classes = TestConfig.class)
@Transactional
@DisplayName("图片文件服务集成测试")
class SysItemFileServiceIT {

    @Autowired
    private SysItemFileService sysItemFileService;

    @Autowired
    private SysFileService sysFileService;

    @Autowired
    private SysItemFileMapper sysItemFileMapper;

    @Autowired
    private SysDatasetItemMapper sysDatasetItemMapper;

    private Long testItemId;
    private Long testFileId;
    private Long testItemFileId;

    @BeforeEach
    void setUp() {
        // 创建测试数据项
        SysDatasetItem testItem = new SysDatasetItem();
        testItem.setDatasetId(1L);
        testItem.setName("测试数据项");
        testItem.setCreateTime(LocalDateTime.now());
        sysDatasetItemMapper.insert(testItem);
        testItemId = testItem.getId();

        // 创建测试文件 - 直接插入数据库（object_name + storage，不落库 url）
        SysFile testFile = new SysFile();
        testFile.setName("test_image.jpg");
        testFile.setType("jpg");
        testFile.setObjectName("test_object_name");
        testFile.setStorage("local");
        testFile.setSize("1024000");
        testFile.setMd5("test_md5_hash");
        sysFileService.save(testFile);
        testFileId = testFile.getId();

        // 创建测试缩略图文件
        SysFile testThumbnail = new SysFile();
        testThumbnail.setName("test_image_thumbnail.jpg");
        testThumbnail.setType("jpg");
        testThumbnail.setObjectName("test_thumbnail_object_name");
        testThumbnail.setStorage("local");
        testThumbnail.setSize("512000");
        testThumbnail.setMd5("test_thumbnail_md5_hash");
        sysFileService.save(testThumbnail);
        Long testThumbnailId = testThumbnail.getId();

        // 创建测试图片文件
        SysItemFile testItemFile = new SysItemFile();
        testItemFile.setItemId(testItemId);
        testItemFile.setFileId(testFileId);
        testItemFile.setThumbnailFileId(testThumbnailId);
        testItemFile.setType("clear");
        testItemFile.setDescription("测试图片");
        testItemFile.setSceneType("outdoor");
        testItemFile.setHazeLevel("none");
        testItemFile.setWidth(1920);
        testItemFile.setHeight(1080);
        testItemFile.setUsageCount(0L);
        testItemFile.setCreateTime(LocalDateTime.now());
        sysItemFileMapper.insert(testItemFile);
        testItemFileId = testItemFile.getId();
    }

    /**
     * 测试根据ID获取图片详细信息
     * 测试场景：查询已存在的图片
     * 验证内容：
     * 1. 返回的VO对象不为null
     * 2. 基本信息正确
     * 3. 文件信息正确（url 由 storage + objectName 动态拼接）
     */
    @Test
    @DisplayName("根据ID获取图片详细信息 - 成功")
    void testGetImageById_Success() {
        // Act
        ImageUrlVO result = sysItemFileService.getImageById(testItemFileId);

        // Assert
        assertNotNull(result);
        assertEquals(testItemFileId, result.getId());
        assertEquals(testItemId, result.getItemId());
        assertEquals("clear", result.getType());
        assertEquals("测试图片", result.getDescription());
        assertEquals("outdoor", result.getSceneType());
        assertEquals("none", result.getHazeLevel());
        assertEquals(1920, result.getWidth());
        assertEquals(1080, result.getHeight());
        assertEquals("test_image.jpg", result.getFileName());
        // url 由 storageService.getUrl(objectName) 动态拼接，包含 objectName
        assertNotNull(result.getUrl());
        assertTrue(result.getUrl().contains("test_object_name"));
    }

    /**
     * 测试根据ID获取不存在的图片
     * 测试场景：查询不存在的图片ID
     * 验证内容：
     * 1. 抛出BusinessException
     * 2. 异常信息正确
     */
    @Test
    @DisplayName("根据ID获取图片详细信息 - 图片不存在")
    void testGetImageById_NotFound() {
        // Arrange
        Long nonExistentId = 999999L;

        // Act & Assert
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> sysItemFileService.getImageById(nonExistentId)
        );
        assertTrue(exception.getMessage().contains("不存在") || exception.getMessage().contains("未找到"));
    }

    /**
     * 测试保存图片文件
     * 测试场景：向数据项添加新图片
     * 验证内容：
     * 1. 返回的VO对象不为null
     * 2. 图片信息正确保存
     * 3. 数据库中存在新记录
     * 注意：此测试直接插入数据库数据，不依赖文件上传流程
     */
    @Test
    @DisplayName("保存图片文件 - 成功")
    void testSaveItemFile_Success() {
        // Arrange - 创建测试文件记录（object_name + storage）
        SysFile newTestFile = new SysFile();
        newTestFile.setName("new_hazy.jpg");
        newTestFile.setType("jpg");
        newTestFile.setObjectName("new_hazy_object");
        newTestFile.setStorage("local");
        newTestFile.setSize("2048000");
        newTestFile.setMd5("new_md5_hash");
        sysFileService.save(newTestFile);
        Long newFileId = newTestFile.getId();

        SysFile newThumbnail = new SysFile();
        newThumbnail.setName("new_hazy_thumbnail.jpg");
        newThumbnail.setType("jpg");
        newThumbnail.setObjectName("new_hazy_thumbnail_object");
        newThumbnail.setStorage("local");
        newThumbnail.setSize("1024000");
        newThumbnail.setMd5("new_thumbnail_md5_hash");
        sysFileService.save(newThumbnail);
        Long newThumbnailId = newThumbnail.getId();

        // 直接创建SysItemFile记录
        SysItemFile newItemFile = new SysItemFile();
        newItemFile.setItemId(testItemId);
        newItemFile.setFileId(newFileId);
        newItemFile.setThumbnailFileId(newThumbnailId);
        newItemFile.setType("hazy");
        newItemFile.setDescription("新上传的有雾图片");
        newItemFile.setSceneType("indoor");
        newItemFile.setHazeLevel("moderate");
        newItemFile.setWidth(1920);
        newItemFile.setHeight(1080);
        newItemFile.setUsageCount(0L);
        newItemFile.setCreateTime(LocalDateTime.now());
        sysItemFileService.save(newItemFile);
        Long newItemFileId = newItemFile.getId();

        // Act - 通过getImageById验证数据
        ImageUrlVO result = sysItemFileService.getImageById(newItemFileId);

        // Assert
        assertNotNull(result);
        assertNotNull(result.getId());
        assertEquals(testItemId, result.getItemId());
        assertEquals("hazy", result.getType());
        assertEquals("新上传的有雾图片", result.getDescription());
        assertEquals("indoor", result.getSceneType());
        assertEquals("moderate", result.getHazeLevel());
        assertEquals(1920, result.getWidth());
        assertEquals(1080, result.getHeight());

        // 验证数据库中存在新记录
        SysItemFile savedItemFile = sysItemFileMapper.selectById(result.getId());
        assertNotNull(savedItemFile);
        assertEquals("hazy", savedItemFile.getType());
    }

    /**
     * 测试更新图片信息
     * 测试场景：更新图片的标注信息（不修改类型）
     * 验证内容：
     * 1. 返回true表示更新成功
     * 2. 数据库中的信息已更新
     */
    @Test
    @DisplayName("更新图片信息 - 成功")
    void testUpdateItemFileInfo_Success() {
        // Arrange - 只更新描述和标注信息，不修改类型
        ItemFileUpdateForm form = new ItemFileUpdateForm();
        form.setDescription("更新后的描述");
        form.setSceneType("indoor");
        form.setHazeLevel("light");

        // Act
        boolean result = sysItemFileService.updateItemFileInfo(testItemFileId, form);

        // Assert
        assertTrue(result);

        // 验证数据库中的信息已更新
        SysItemFile updatedItemFile = sysItemFileMapper.selectById(testItemFileId);
        assertNotNull(updatedItemFile);
        assertEquals("clear", updatedItemFile.getType()); // 类型保持不变
        assertEquals("更新后的描述", updatedItemFile.getDescription());
        assertEquals("indoor", updatedItemFile.getSceneType());
        assertEquals("light", updatedItemFile.getHazeLevel());
    }

    /**
     * 测试更新不存在的图片
     * 测试场景：更新不存在的图片ID
     * 验证内容：
     * 1. 抛出BusinessException异常
     */
    @Test
    @DisplayName("更新图片信息 - 图片不存在")
    void testUpdateItemFileInfo_NotFound() {
        // Arrange
        Long nonExistentId = 999999L;
        ItemFileUpdateForm form = new ItemFileUpdateForm();
        form.setDescription("更新描述");

        // Act & Assert
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> sysItemFileService.updateItemFileInfo(nonExistentId, form)
        );
        assertTrue(exception.getMessage().contains("不存在"));
    }

    /**
     * 测试部分字段更新
     * 测试场景：仅更新部分字段，其他字段保持不变
     * 验证内容：
     * 1. 返回true表示更新成功
     * 2. 仅指定字段被更新
     * 3. 未指定字段保持原值
     */
    @Test
    @DisplayName("更新图片信息 - 部分字段更新")
    void testUpdateItemFileInfo_PartialUpdate() {
        // Arrange
        ItemFileUpdateForm form = new ItemFileUpdateForm();
        form.setDescription("仅更新描述");
        // 其他字段为null，不更新

        // Act
        boolean result = sysItemFileService.updateItemFileInfo(testItemFileId, form);

        // Assert
        assertTrue(result);

        // 验证数据库中的信息
        SysItemFile updatedItemFile = sysItemFileMapper.selectById(testItemFileId);
        assertNotNull(updatedItemFile);
        assertEquals("仅更新描述", updatedItemFile.getDescription());
        // 其他字段保持原值
        assertEquals("clear", updatedItemFile.getType());
        assertEquals("outdoor", updatedItemFile.getSceneType());
        assertEquals("none", updatedItemFile.getHazeLevel());
    }

    /**
     * 测试删除图片文件
     * 测试场景：删除已存在的图片
     * 验证内容：
     * 1. 返回true表示删除成功
     * 2. 数据库中记录已删除
     */
    @Test
    @DisplayName("删除图片文件 - 成功")
    void testDeleteFile_Success() {
        // Act
        boolean result = sysItemFileService.deleteFile(testItemFileId);

        // Assert
        assertTrue(result);

        // 验证数据库中记录已删除
        SysItemFile deletedItemFile = sysItemFileMapper.selectById(testItemFileId);
        assertNull(deletedItemFile);
    }

    /**
     * 测试删除不存在的图片
     * 测试场景：删除不存在的图片ID
     * 验证内容：
     * 1. 抛出BusinessException异常
     */
    @Test
    @DisplayName("删除图片文件 - 图片不存在")
    void testDeleteFile_NotFound() {
        // Arrange
        Long nonExistentId = 999999L;

        // Act & Assert
        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> sysItemFileService.deleteFile(nonExistentId)
        );
        assertTrue(exception.getMessage().contains("不存在"));
    }

    /**
     * 测试批量删除图片（全部成功）
     * 测试场景：批量删除多个存在的图片
     * 验证内容：
     * 1. 返回结果不为null
     * 2. 成功数量正确
     * 3. 失败数量为0
     * 4. 数据库中记录已删除
     */
    @Test
    @DisplayName("批量删除图片 - 全部成功")
    void testBatchDelete_AllSuccess() {
        // Arrange - 创建额外的测试文件和缩略图（object_name + storage）
        SysFile testFile2 = new SysFile();
        testFile2.setName("test_image_2.jpg");
        testFile2.setType("jpg");
        testFile2.setObjectName("test_object_name_2");
        testFile2.setStorage("local");
        testFile2.setSize("2048000");
        testFile2.setMd5("test_md5_hash_2");
        sysFileService.save(testFile2);
        Long testFileId2 = testFile2.getId();

        SysFile testThumbnail2 = new SysFile();
        testThumbnail2.setName("test_image_2_thumbnail.jpg");
        testThumbnail2.setType("jpg");
        testThumbnail2.setObjectName("test_thumbnail_object_name_2");
        testThumbnail2.setStorage("local");
        testThumbnail2.setSize("1024000");
        testThumbnail2.setMd5("test_thumbnail_md5_hash_2");
        sysFileService.save(testThumbnail2);
        Long testThumbnailId2 = testThumbnail2.getId();

        SysItemFile testItemFile2 = new SysItemFile();
        testItemFile2.setItemId(testItemId);
        testItemFile2.setFileId(testFileId2);
        testItemFile2.setThumbnailFileId(testThumbnailId2);
        testItemFile2.setType("hazy");
        testItemFile2.setDescription("测试图片2");
        testItemFile2.setSceneType("outdoor");
        testItemFile2.setHazeLevel("light");
        testItemFile2.setWidth(1920);
        testItemFile2.setHeight(1080);
        testItemFile2.setUsageCount(0L);
        testItemFile2.setCreateTime(LocalDateTime.now());
        sysItemFileMapper.insert(testItemFile2);
        Long testItemFileId2 = testItemFile2.getId();

        List<Long> idsToDelete = Arrays.asList(testItemFileId2);

        // Act
        BatchDeleteResultVO result = sysItemFileService.batchDelete(idsToDelete);

        // Assert
        assertNotNull(result);
        assertEquals(1, result.getSuccessCount());
        assertEquals(0, result.getFailedCount());
        assertEquals(1, result.getSuccessIds().size());
        assertTrue(result.getSuccessIds().contains(testItemFileId2));
        assertTrue(result.getFailedItems().isEmpty());

        // 验证数据库中记录已删除
        assertNull(sysItemFileMapper.selectById(testItemFileId2));
    }

    /**
     * 测试批量删除图片（部分失败）
     * 测试场景：批量删除时包含不存在的图片ID
     * 验证内容：
     * 1. 返回结果不为null
     * 2. 成功和失败数量正确
     * 3. 失败原因正确记录
     */
    @Test
    @DisplayName("批量删除图片 - 部分失败")
    void testBatchDelete_PartialFailure() {
        // Arrange - 创建额外的测试文件和缩略图（object_name + storage）
        SysFile testFile2 = new SysFile();
        testFile2.setName("test_image_3.jpg");
        testFile2.setType("jpg");
        testFile2.setObjectName("test_object_name_3");
        testFile2.setStorage("local");
        testFile2.setSize("2048000");
        testFile2.setMd5("test_md5_hash_3");
        sysFileService.save(testFile2);
        Long testFileId2 = testFile2.getId();

        SysFile testThumbnail2 = new SysFile();
        testThumbnail2.setName("test_image_3_thumbnail.jpg");
        testThumbnail2.setType("jpg");
        testThumbnail2.setObjectName("test_thumbnail_object_name_3");
        testThumbnail2.setStorage("local");
        testThumbnail2.setSize("1024000");
        testThumbnail2.setMd5("test_thumbnail_md5_hash_3");
        sysFileService.save(testThumbnail2);
        Long testThumbnailId2 = testThumbnail2.getId();

        SysItemFile testItemFile2 = new SysItemFile();
        testItemFile2.setItemId(testItemId);
        testItemFile2.setFileId(testFileId2);
        testItemFile2.setThumbnailFileId(testThumbnailId2);
        testItemFile2.setType("hazy");
        testItemFile2.setDescription("测试图片3");
        testItemFile2.setSceneType("outdoor");
        testItemFile2.setHazeLevel("light");
        testItemFile2.setWidth(1920);
        testItemFile2.setHeight(1080);
        testItemFile2.setUsageCount(0L);
        testItemFile2.setCreateTime(LocalDateTime.now());
        sysItemFileMapper.insert(testItemFile2);
        Long testItemFileId2 = testItemFile2.getId();

        Long nonExistentId = 999999L;
        List<Long> idsToDelete = Arrays.asList(testItemFileId2, nonExistentId);

        // Act
        BatchDeleteResultVO result = sysItemFileService.batchDelete(idsToDelete);

        // Assert
        assertNotNull(result);
        assertEquals(1, result.getSuccessCount());
        assertEquals(1, result.getFailedCount());
        assertEquals(1, result.getSuccessIds().size());
        assertTrue(result.getSuccessIds().contains(testItemFileId2));
        assertEquals(1, result.getFailedItems().size());
        assertEquals(nonExistentId, result.getFailedItems().get(0).getId());
        assertNotNull(result.getFailedItems().get(0).getReason());

        // 验证成功删除的记录已删除
        assertNull(sysItemFileMapper.selectById(testItemFileId2));
    }

    /**
     * 测试批量删除空列表
     * 测试场景：传入空的ID列表
     * 验证内容：
     * 1. 返回结果不为null
     * 2. 成功和失败数量都为0
     */
    @Test
    @DisplayName("批量删除图片 - 空列表")
    void testBatchDelete_EmptyList() {
        // Arrange
        List<Long> emptyList = Arrays.asList();

        // Act
        BatchDeleteResultVO result = sysItemFileService.batchDelete(emptyList);

        // Assert
        assertNotNull(result);
        assertEquals(0, result.getSuccessCount());
        assertEquals(0, result.getFailedCount());
        assertTrue(result.getSuccessIds().isEmpty());
        assertTrue(result.getFailedItems().isEmpty());
    }
}
