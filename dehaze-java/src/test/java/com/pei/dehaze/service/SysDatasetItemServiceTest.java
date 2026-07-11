package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.mapper.SysDatasetItemMapper;
import com.pei.dehaze.mapper.SysDatasetMapper;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.query.DatasetItemQuery;
import com.pei.dehaze.model.vo.BatchActionFailureDetailVO;
import com.pei.dehaze.model.vo.BatchOperationResultVO;
import com.pei.dehaze.model.vo.DatasetItemVO;
import com.pei.dehaze.model.vo.ImageUrlVO;
import com.pei.dehaze.service.impl.DatasetOperationServiceImpl;
import com.pei.dehaze.service.impl.SysDatasetItemServiceImpl;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;
import static org.mockito.Mockito.spy;

/**
 * 数据项服务单元测试
 * 测试目的：验证SysDatasetItemService的业务逻辑正确性
 * 测试范围：
 * 1. 数据项创建业务逻辑
 * 2. 数据项删除及级联删除逻辑
 * 3. 数据项更新及场景类型级联更新
 * 4. 分页查询及叶子节点获取逻辑
 * 5. 批量删除及异常处理
 * 6. 数据项详情查询及图片分类排序
 * <p>
 * 注意：本测试专注于Service层业务逻辑，不重复Controller已覆盖的集成测试场景
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("数据项服务测试")
class SysDatasetItemServiceTest {

    @Mock
    private SysDatasetItemMapper sysDatasetItemMapper;

    @Mock
    private SysDatasetMapper sysDatasetMapper;

    @Mock
    private SysItemFileService sysItemFileService;

    @Mock
    private SysFileService sysFileService;

    @Mock
    private SysDatasetService sysDatasetService;

    private SysDatasetItemServiceImpl sysDatasetItemService;

    private DatasetOperationServiceImpl datasetOperationService;

    private SysDatasetItem testDatasetItem;
    private List<SysItemFile> testItemFiles;

    @BeforeEach
    void setUp() throws NoSuchFieldException, IllegalAccessException {
        // 初始化服务实例并使用spy来mock removeById方法
        sysDatasetItemService = spy(new SysDatasetItemServiceImpl());

        // 使用反射设置baseMapper
        Field baseMapperField = SysDatasetItemServiceImpl.class.getSuperclass().getDeclaredField("baseMapper");
        baseMapperField.setAccessible(true);
        baseMapperField.set(sysDatasetItemService, sysDatasetItemMapper);

        // 使用反射设置sysDatasetMapper
        Field sysDatasetMapperField = SysDatasetItemServiceImpl.class.getDeclaredField("sysDatasetMapper");
        sysDatasetMapperField.setAccessible(true);
        sysDatasetMapperField.set(sysDatasetItemService, sysDatasetMapper);

        // 使用反射设置sysItemFileService
        Field sysItemFileServiceField = SysDatasetItemServiceImpl.class.getDeclaredField("sysItemFileService");
        sysItemFileServiceField.setAccessible(true);
        sysItemFileServiceField.set(sysDatasetItemService, sysItemFileService);

        // 创建DatasetOperationServiceImpl并注入依赖（batchDelete已迁移至此服务）
        datasetOperationService = spy(new DatasetOperationServiceImpl());

        Field datasetOpDatasetServiceField = DatasetOperationServiceImpl.class.getDeclaredField("sysDatasetService");
        datasetOpDatasetServiceField.setAccessible(true);
        datasetOpDatasetServiceField.set(datasetOperationService, sysDatasetService);

        Field datasetOpItemServiceField = DatasetOperationServiceImpl.class.getDeclaredField("sysDatasetItemService");
        datasetOpItemServiceField.setAccessible(true);
        datasetOpItemServiceField.set(datasetOperationService, sysDatasetItemService);

        Field datasetOpItemFileServiceField = DatasetOperationServiceImpl.class.getDeclaredField("sysItemFileService");
        datasetOpItemFileServiceField.setAccessible(true);
        datasetOpItemFileServiceField.set(datasetOperationService, sysItemFileService);

        // Mock removeById方法调用mapper的deleteById
        lenient().doAnswer(invocation -> {
            Long id = invocation.getArgument(0);
            return sysDatasetItemMapper.deleteById(id) > 0;
        }).when(sysDatasetItemService).removeById(anyLong());

        // 初始化测试数据
        testDatasetItem = new SysDatasetItem();
        testDatasetItem.setId(1L);
        testDatasetItem.setDatasetId(10L);
        testDatasetItem.setName("测试数据项_001");
        testDatasetItem.setCreateTime(LocalDateTime.now());
        testDatasetItem.setUpdateTime(LocalDateTime.now());

        // 初始化测试图片数据
        testItemFiles = new ArrayList<>();

        SysItemFile clearFile = new SysItemFile();
        clearFile.setId(101L);
        clearFile.setItemId(1L);
        clearFile.setType("clear");
        clearFile.setFileId(1001L);
        clearFile.setWidth(1920);
        clearFile.setHeight(1080);
        clearFile.setSceneType("indoor");
        testItemFiles.add(clearFile);

        SysItemFile hazyFile1 = new SysItemFile();
        hazyFile1.setId(102L);
        hazyFile1.setItemId(1L);
        hazyFile1.setType("hazy");
        hazyFile1.setFileId(1002L);
        hazyFile1.setHazeLevel("light");
        hazyFile1.setWidth(1920);
        hazyFile1.setHeight(1080);
        hazyFile1.setSceneType("outdoor");
        testItemFiles.add(hazyFile1);

        SysItemFile hazyFile2 = new SysItemFile();
        hazyFile2.setId(103L);
        hazyFile2.setItemId(1L);
        hazyFile2.setType("hazy");
        hazyFile2.setFileId(1003L);
        hazyFile2.setHazeLevel("medium");
        hazyFile2.setWidth(1920);
        hazyFile2.setHeight(1080);
        hazyFile2.setSceneType("outdoor");
        testItemFiles.add(hazyFile2);

        SysItemFile hazyFile3 = new SysItemFile();
        hazyFile3.setId(104L);
        hazyFile3.setItemId(1L);
        hazyFile3.setType("hazy");
        hazyFile3.setFileId(1004L);
        hazyFile3.setHazeLevel("heavy");
        hazyFile3.setWidth(1920);
        hazyFile3.setHeight(1080);
        hazyFile3.setSceneType("outdoor");
        testItemFiles.add(hazyFile3);
    }

    // ==================== 批量删除测试 ====================

    /**
     * 测试批量删除数据项（全部成功）
     * 测试场景：批量删除多个数据项，全部成功
     * 注意：批量删除已迁移至DatasetOperationService.batchDeleteDatasetItemsCascadeWithResult
     * 验证内容：
     * 1. 返回正确的成功和失败数量
     * 2. 失败详情为空
     * 3. 所有数据项都被删除
     */
    @Test
    @DisplayName("batchDeleteDatasetItemsCascadeWithResult - 批量删除全部成功")
    void testBatchDeleteDatasetItems_AllSuccess() {
        // Arrange
        List<Long> itemIds = Arrays.asList(1L, 2L, 3L);

        // mock: 数据项存在
        doReturn(testDatasetItem).when(sysDatasetItemService).getById(anyLong());
        // mock: 无关联文件
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(Collections.emptyList());
        // mock: removeById成功（通过mapper的deleteById返回1）
        when(sysDatasetItemMapper.deleteById(anyLong())).thenReturn(1);

        // Act
        BatchOperationResultVO result = datasetOperationService.batchDeleteDatasetItemsCascadeWithResult(itemIds);

        // Assert
        assertNotNull(result);
        assertEquals(3, result.getSuccessCount());
        assertEquals(0, result.getFailedCount());
        assertTrue(result.getFailureDetails() == null || result.getFailureDetails().isEmpty());
        // 验证每个ID都被删除了
        verify(sysDatasetItemService, times(1)).removeById(1L);
        verify(sysDatasetItemService, times(1)).removeById(2L);
        verify(sysDatasetItemService, times(1)).removeById(3L);
    }

    /**
     * 测试批量删除数据项（部分失败）
     * 测试场景：批量删除多个数据项，部分成功部分失败
     * 验证内容：
     * 1. 返回正确的成功和失败数量
     * 2. 失败详情包含正确的信息
     * 3. 成功的数据项被删除
     */
    @Test
    @DisplayName("batchDeleteDatasetItemsCascadeWithResult - 批量删除部分失败")
    void testBatchDeleteDatasetItems_PartialFailure() {
        // Arrange
        List<Long> itemIds = Arrays.asList(1L, 2L, 3L);

        // mock: 数据项存在
        doReturn(testDatasetItem).when(sysDatasetItemService).getById(anyLong());
        // mock: 无关联文件
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(Collections.emptyList());
        // 第一个和第三个成功，第二个失败
        when(sysDatasetItemMapper.deleteById(1L)).thenReturn(1);
        when(sysDatasetItemMapper.deleteById(2L)).thenThrow(new RuntimeException("数据库连接失败"));
        when(sysDatasetItemMapper.deleteById(3L)).thenReturn(1);

        // Act
        BatchOperationResultVO result = datasetOperationService.batchDeleteDatasetItemsCascadeWithResult(itemIds);

        // Assert
        assertNotNull(result);
        assertEquals(2, result.getSuccessCount());
        assertEquals(1, result.getFailedCount());
        assertNotNull(result.getFailureDetails());
        assertEquals(1, result.getFailureDetails().size());

        BatchActionFailureDetailVO failureDetail = result.getFailureDetails().get(0);
        assertEquals("2", failureDetail.getIdentifier());
        assertEquals("数据库连接失败", failureDetail.getReason());
    }

    /**
     * 创建测试用的SysDatasetItem
     */
    private SysDatasetItem createTestDatasetItem(Long id) {
        SysDatasetItem item = new SysDatasetItem();
        item.setId(id);
        item.setDatasetId(10L);
        item.setName("测试数据项_" + id);
        item.setCreateTime(LocalDateTime.now());
        item.setUpdateTime(LocalDateTime.now());
        return item;
    }

    /**
     * 测试批量删除数据项（空列表）
     * 测试场景：传入空的ID列表
     * 验证内容：
     * 1. 返回正确的结果（成功和失败都为0）
     * 2. 不调用任何删除方法
     */
    @Test
    @DisplayName("batchDeleteDatasetItemsCascadeWithResult - 空列表")
    void testBatchDeleteDatasetItems_EmptyList() {
        // Arrange
        List<Long> itemIds = Collections.emptyList();

        // Act
        BatchOperationResultVO result = datasetOperationService.batchDeleteDatasetItemsCascadeWithResult(itemIds);

        // Assert
        assertNotNull(result);
        assertEquals(0, result.getSuccessCount());
        assertEquals(0, result.getFailedCount());
        assertEquals("没有需要删除的数据项", result.getMessage());
        verify(sysDatasetItemMapper, never()).deleteById(anyLong());
    }

    /**
     * 测试批量删除数据项（null列表）
     * 测试场景：传入null列表
     * 验证内容：
     * 1. 返回正确的结果
     * 2. 不抛出异常
     */
    @Test
    @DisplayName("batchDeleteDatasetItemsCascadeWithResult - null列表")
    void testBatchDeleteDatasetItems_NullList() {
        // Arrange
        List<Long> itemIds = null;

        // Act
        BatchOperationResultVO result = datasetOperationService.batchDeleteDatasetItemsCascadeWithResult(itemIds);

        // Assert
        assertNotNull(result);
        assertEquals(0, result.getSuccessCount());
        assertEquals(0, result.getFailedCount());
        assertEquals("没有需要删除的数据项", result.getMessage());
    }

    // ==================== 删除数据项测试 ====================

    /**
     * 测试删除数据项（有图片）
     * 测试场景：删除包含图片的数据项
     * 注意：重构后deleteDatasetItem只删除数据项记录，不删除图片
     * 图片删除由DatasetOperationService.deleteDatasetItemCascade处理
     * 验证内容：
     * 1. 只删除数据项记录
     * 2. 不删除图片
     */
    @Test
    @DisplayName("deleteDatasetItem - 删除数据项（有图片）")
    void testDeleteDatasetItem_WithImages() {
        // Arrange
        Long itemId = 1L;

        when(sysDatasetItemMapper.deleteById(itemId)).thenReturn(1);

        // Act
        sysDatasetItemService.deleteDatasetItem(itemId);

        // Assert
        verify(sysDatasetItemMapper, times(1)).deleteById(itemId);
        // 重构后不再删除图片
        verify(sysItemFileService, never()).list(any(com.baomidou.mybatisplus.core.conditions.Wrapper.class));
        verify(sysItemFileService, never()).deleteFile(anyLong());
    }

    /**
     * 测试删除数据项（无图片）
     * 测试场景：删除不包含图片的数据项
     * 验证内容：
     * 1. 数据项被删除
     * 2. 不会调用图片相关方法
     */
    @Test
    @DisplayName("deleteDatasetItem - 删除数据项（无图片）")
    void testDeleteDatasetItem_WithoutImages() {
        // Arrange
        Long itemId = 1L;

        when(sysDatasetItemMapper.deleteById(itemId)).thenReturn(1);

        // Act
        sysDatasetItemService.deleteDatasetItem(itemId);

        // Assert
        verify(sysDatasetItemMapper, times(1)).deleteById(itemId);
        verify(sysItemFileService, never()).list(any(com.baomidou.mybatisplus.core.conditions.Wrapper.class));
        verify(sysItemFileService, never()).deleteFile(anyLong());
    }

    /**
     * 测试删除不存在的数据项
     * 测试场景：删除不存在的数据项ID
     * 验证内容：
     * 1. 方法正常执行
     * 2. 返回值为0（未删除任何记录）
     */
    @Test
    @DisplayName("deleteDatasetItem - 删除不存在的数据项")
    void testDeleteDatasetItem_NotExists() {
        // Arrange
        Long itemId = 999L;

        when(sysDatasetItemMapper.deleteById(itemId)).thenReturn(0);

        // Act
        sysDatasetItemService.deleteDatasetItem(itemId);

        // Assert
        verify(sysDatasetItemMapper, times(1)).deleteById(itemId);
    }

    // ==================== 创建数据项测试 ====================

    /**
     * 测试创建数据项（仅指定数据集ID）
     * 测试场景：使用基础创建方法创建数据项
     * 验证内容：
     * 1. 数据项成功创建
     * 2. 数据集ID正确设置
     * 3. 名称字段为null（因为未指定）
     */
    @Test
    @DisplayName("createDatasetItem - 仅指定数据集ID创建数据项")
    void testCreateDatasetItem_WithDatasetIdOnly() {
        // Arrange
        Long datasetId = 10L;
        when(sysDatasetItemMapper.insert(any(SysDatasetItem.class))).thenAnswer(invocation -> {
            SysDatasetItem item = invocation.getArgument(0);
            item.setId(1L);
            return 1;
        });

        // Act
        SysDatasetItem result = sysDatasetItemService.createDatasetItem(datasetId);

        // Assert
        assertNotNull(result);
        assertEquals(datasetId, result.getDatasetId());
        assertNull(result.getName());
        assertNotNull(result.getId());
        verify(sysDatasetItemMapper, times(1)).insert(any(SysDatasetItem.class));
    }

    /**
     * 测试创建数据项（指定数据集ID和名称）
     * 测试场景：使用完整参数创建数据项
     * 验证内容：
     * 1. 数据项成功创建
     * 2. 数据集ID和名称都正确设置
     */
    @Test
    @DisplayName("createDatasetItem - 指定数据集ID和名称创建数据项")
    void testCreateDatasetItem_WithDatasetIdAndName() {
        // Arrange
        Long datasetId = 10L;
        String itemName = "测试数据项";
        when(sysDatasetItemMapper.insert(any(SysDatasetItem.class))).thenAnswer(invocation -> {
            SysDatasetItem item = invocation.getArgument(0);
            item.setId(1L);
            return 1;
        });

        // Act
        SysDatasetItem result = sysDatasetItemService.createDatasetItem(datasetId, itemName);

        // Assert
        assertNotNull(result);
        assertEquals(datasetId, result.getDatasetId());
        assertEquals(itemName, result.getName());
        assertNotNull(result.getId());
        verify(sysDatasetItemMapper, times(1)).insert(any(SysDatasetItem.class));
    }

    /**
     * 测试创建数据项并返回完整VO
     * 测试场景：创建数据项后立即获取完整信息
     * 验证内容：
     * 1. 数据项创建成功
     * 2. 返回包含完整信息的VO对象
     */
    @Test
    @DisplayName("createAndReturnDatasetItem - 创建数据项并返回VO")
    void testCreateAndReturnDatasetItem() {
        // Arrange
        Long datasetId = 10L;
        String itemName = "测试数据项";

        when(sysDatasetItemMapper.insert(any(SysDatasetItem.class))).thenAnswer(invocation -> {
            SysDatasetItem item = invocation.getArgument(0);
            item.setId(1L);
            item.setName(itemName); // 确保名称正确
            return 1;
        });

        // 创建一个新的item用于返回，而不是使用预设的testDatasetItem
        SysDatasetItem newItem = new SysDatasetItem();
        newItem.setId(1L);
        newItem.setDatasetId(datasetId);
        newItem.setName(itemName);
        newItem.setCreateTime(LocalDateTime.now());
        newItem.setUpdateTime(LocalDateTime.now());

        when(sysDatasetItemMapper.selectById(1L)).thenReturn(newItem);
        when(sysItemFileService.getImageUrlVOs(1L)).thenReturn(Collections.emptyList());

        // Act
        DatasetItemVO result = sysDatasetItemService.createAndReturnDatasetItem(datasetId, itemName);

        // Assert
        assertNotNull(result);
        assertEquals(1L, result.getId());
        assertEquals(datasetId, result.getDatasetId());
        assertEquals(itemName, result.getName());
        verify(sysDatasetItemMapper, times(1)).insert(any(SysDatasetItem.class));
        verify(sysDatasetItemMapper, times(1)).selectById(1L);
    }

    // ==================== 更新数据项测试 ====================

    /**
     * 测试更新数据项（仅更新名称）
     * 测试场景：只更新数据项名称，不修改场景类型
     * 验证内容：
     * 1. 数据项名称更新成功
     * 2. 场景类型不会被更新
     * 3. 不会触发图片的级联更新
     */
    @Test
    @DisplayName("updateDatasetItem - 仅更新数据项名称")
    void testUpdateDatasetItem_NameOnly() {
        // Arrange
        Long itemId = 1L;
        String newName = "更新后的名称";

        when(sysDatasetItemMapper.selectById(itemId)).thenReturn(testDatasetItem);
        when(sysDatasetItemMapper.updateById(any(SysDatasetItem.class))).thenReturn(1);

        // Act
        sysDatasetItemService.updateDatasetItem(itemId, newName);

        // Assert
        verify(sysDatasetItemMapper, times(1)).selectById(itemId);
        verify(sysDatasetItemMapper, times(1)).updateById(any(SysDatasetItem.class));
    }

    /**
     * 测试更新数据项并返回VO（仅更新名称）
     * 测试场景：更新名称并返回完整VO
     * 验证内容：
     * 1. 名称更新成功
     * 2. 返回包含更新后信息的VO
     */
    @Test
    @DisplayName("updateAndReturnDatasetItem - 仅更新名称并返回VO")
    void testUpdateAndReturnDatasetItem_NameOnly() {
        // Arrange
        Long itemId = 1L;
        String newName = "更新后的名称";
        String sceneType = null;

        when(sysDatasetItemMapper.selectById(itemId)).thenReturn(testDatasetItem);
        when(sysDatasetItemMapper.updateById(any(SysDatasetItem.class))).thenReturn(1);

        List<ImageUrlVO> imageUrlVOs = createMockImageUrlVOs();
        when(sysItemFileService.getImageUrlVOs(itemId)).thenReturn(imageUrlVOs);

        // Act
        DatasetItemVO result = sysDatasetItemService.updateAndReturnDatasetItem(itemId, newName, sceneType);

        // Assert
        assertNotNull(result);
        assertEquals(itemId, result.getId());
        assertEquals(newName, result.getName());
        verify(sysDatasetItemMapper, times(1)).updateById(any(SysDatasetItem.class));
        verify(sysItemFileService, never()).updateBatchById(any());
    }

    /**
     * 测试更新数据项并返回VO（更新场景类型）
     * 测试场景：更新场景类型，验证级联更新逻辑
     * 验证内容：
     * 1. 数据项场景类型更新成功
     * 2. 关联的所有图片场景类型同步更新
     * 3. 调用了批量更新方法
     */
    @Test
    @DisplayName("updateAndReturnDatasetItem - 更新场景类型并级联更新图片")
    void testUpdateAndReturnDatasetItem_WithSceneType() {
        // Arrange
        Long itemId = 1L;
        String newName = "更新后的名称";
        String sceneType = "indoor";

        when(sysDatasetItemMapper.selectById(itemId)).thenReturn(testDatasetItem);
        when(sysDatasetItemMapper.updateById(any(SysDatasetItem.class))).thenReturn(1);
        when(sysItemFileService.list(any(com.baomidou.mybatisplus.core.conditions.Wrapper.class))).thenReturn(testItemFiles);
        when(sysItemFileService.updateBatchById(anyList())).thenReturn(true);

        List<ImageUrlVO> imageUrlVOs = createMockImageUrlVOs();
        when(sysItemFileService.getImageUrlVOs(itemId)).thenReturn(imageUrlVOs);

        // Act
        DatasetItemVO result = sysDatasetItemService.updateAndReturnDatasetItem(itemId, newName, sceneType);

        // Assert
        assertNotNull(result);
        assertEquals(itemId, result.getId());
        assertEquals(sceneType, result.getSceneType());
        verify(sysDatasetItemMapper, times(1)).updateById(any(SysDatasetItem.class));
        verify(sysItemFileService, times(1)).list(any(com.baomidou.mybatisplus.core.conditions.Wrapper.class));
        verify(sysItemFileService, times(1)).updateBatchById(anyList());

        // 验证所有图片的场景类型都被更新
        for (SysItemFile itemFile : testItemFiles) {
            assertEquals(sceneType, itemFile.getSceneType());
        }
    }

    /**
     * 测试更新不存在的数据项
     * 测试场景：更新不存在的数据项ID
     * 验证内容：
     * 1. 抛出RuntimeException
     * 2. 异常消息正确
     */
    @Test
    @DisplayName("updateAndReturnDatasetItem - 更新不存在的数据项抛出异常")
    void testUpdateAndReturnDatasetItem_NotExists() {
        // Arrange
        Long itemId = 999L;
        String newName = "更新后的名称";
        String sceneType = "indoor";

        when(sysDatasetItemMapper.selectById(itemId)).thenReturn(null);

        // Act & Assert
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            sysDatasetItemService.updateAndReturnDatasetItem(itemId, newName, sceneType);
        });

        assertEquals("数据项不存在", exception.getMessage());
        verify(sysDatasetItemMapper, times(1)).selectById(itemId);
        verify(sysDatasetItemMapper, never()).updateById(any(SysDatasetItem.class));
    }

    /**
     * 测试更新数据项（名称为null）
     * 测试场景：只更新场景类型，不更新名称
     * 验证内容：
     * 1. 数据项不更新名称
     * 2. 场景类型正常更新
     * 3. 不会调用updateById更新名称
     */
    @Test
    @DisplayName("updateAndReturnDatasetItem - 名称为null时不更新名称")
    void testUpdateAndReturnDatasetItem_NullName() {
        // Arrange
        Long itemId = 1L;
        String newName = null;
        String sceneType = "indoor";

        when(sysDatasetItemMapper.selectById(itemId)).thenReturn(testDatasetItem);
        when(sysItemFileService.list(any(com.baomidou.mybatisplus.core.conditions.Wrapper.class))).thenReturn(testItemFiles);
        when(sysItemFileService.updateBatchById(anyList())).thenReturn(true);

        List<ImageUrlVO> imageUrlVOs = createMockImageUrlVOs();
        when(sysItemFileService.getImageUrlVOs(itemId)).thenReturn(imageUrlVOs);

        // Act
        DatasetItemVO result = sysDatasetItemService.updateAndReturnDatasetItem(itemId, newName, sceneType);

        // Assert
        assertNotNull(result);
        assertEquals(sceneType, result.getSceneType());
        assertEquals("测试数据项_001", result.getName());
        verify(sysDatasetItemMapper, never()).updateById(any(SysDatasetItem.class));
        verify(sysItemFileService, times(1)).updateBatchById(anyList());
    }

    // ==================== 获取数据项详情测试 ====================

    /**
     * 测试获取数据项详情（有清晰图和有雾图）
     * 测试场景：获取包含清晰图和有雾图的数据项详情
     * 验证内容：
     * 1. 返回完整的VO对象
     * 2. 清晰图正确识别
     * 3. 有雾图正确识别并按雾霾程度排序
     */
    @Test
    @DisplayName("getDatasetItem - 获取数据项详情（有清晰图和有雾图）")
    void testGetDatasetItem_WithClearAndHazyImages() {
        // Arrange
        Long itemId = 1L;

        when(sysDatasetItemMapper.selectById(itemId)).thenReturn(testDatasetItem);
        List<ImageUrlVO> imageUrlVOs = createMockImageUrlVOs();
        when(sysItemFileService.getImageUrlVOs(itemId)).thenReturn(imageUrlVOs);

        // Act
        DatasetItemVO result = sysDatasetItemService.getDatasetItem(itemId);

        // Assert
        assertNotNull(result);
        assertEquals(itemId, result.getId());
        assertEquals(testDatasetItem.getName(), result.getName());
        assertEquals(testDatasetItem.getDatasetId(), result.getDatasetId());
        assertEquals(4, result.getImageCount());

        // 验证清晰图
        assertNotNull(result.getClearImage());
        assertEquals("clear", result.getClearImage().getType());

        // 验证有雾图
        assertNotNull(result.getHazyImages());
        assertEquals(3, result.getHazyImages().size());

        // 验证有雾图按雾霾程度排序（light < medium < heavy）
        assertEquals("light", result.getHazyImages().get(0).getHazeLevel());
        assertEquals("medium", result.getHazyImages().get(1).getHazeLevel());
        assertEquals("heavy", result.getHazyImages().get(2).getHazeLevel());

        // 验证场景类型从清晰图获取
        assertEquals("indoor", result.getSceneType());
    }

    /**
     * 测试获取数据项详情（只有有雾图）
     * 测试场景：数据项只有有雾图，没有清晰图
     * 验证内容：
     * 1. 返回VO对象
     * 2. 清晰图为null
     * 3. 有雾图列表正确
     */
    @Test
    @DisplayName("getDatasetItem - 只有有雾图的数据项")
    void testGetDatasetItem_WithHazyImagesOnly() {
        // Arrange
        Long itemId = 1L;

        when(sysDatasetItemMapper.selectById(itemId)).thenReturn(testDatasetItem);
        List<ImageUrlVO> imageUrlVOs = new ArrayList<>();

        ImageUrlVO hazyImage = new ImageUrlVO();
        hazyImage.setId(102L);
        hazyImage.setType("hazy");
        hazyImage.setHazeLevel("light");
        hazyImage.setSceneType("indoor");
        imageUrlVOs.add(hazyImage);

        when(sysItemFileService.getImageUrlVOs(itemId)).thenReturn(imageUrlVOs);

        // Act
        DatasetItemVO result = sysDatasetItemService.getDatasetItem(itemId);

        // Assert
        assertNotNull(result);
        assertNull(result.getClearImage());
        assertNotNull(result.getHazyImages());
        assertEquals(1, result.getHazyImages().size());

        // 场景类型从有雾图获取
        assertEquals("indoor", result.getSceneType());
    }

    /**
     * 测试获取数据项详情（无图片）
     * 测试场景：数据项没有任何图片
     * 验证内容：
     * 1. 返回基本信息的VO
     * 2. 图片数量为0
     * 3. 清晰图和有雾图都为空
     */
    @Test
    @DisplayName("getDatasetItem - 无图片的数据项")
    void testGetDatasetItem_NoImages() {
        // Arrange
        Long itemId = 1L;

        when(sysDatasetItemMapper.selectById(itemId)).thenReturn(testDatasetItem);
        when(sysItemFileService.getImageUrlVOs(itemId)).thenReturn(Collections.emptyList());

        // Act
        DatasetItemVO result = sysDatasetItemService.getDatasetItem(itemId);

        // Assert
        assertNotNull(result);
        assertEquals(itemId, result.getId());
        assertEquals(0, result.getImageCount());
        assertNull(result.getClearImage());
        assertNotNull(result.getHazyImages());
        assertTrue(result.getHazyImages().isEmpty());
        assertNull(result.getSceneType());
    }

    /**
     * 测试获取数据项详情（有雾图未排序）
     * 测试场景：有雾图乱序传入，验证排序逻辑
     * 验证内容：
     * 1. 有雾图按雾霾程度正确排序
     */
    @Test
    @DisplayName("getDatasetItem - 有雾图按雾霾程度排序")
    void testGetDatasetItem_HazyImageSorting() {
        // Arrange
        Long itemId = 1L;

        when(sysDatasetItemMapper.selectById(itemId)).thenReturn(testDatasetItem);
        List<ImageUrlVO> imageUrlVOs = new ArrayList<>();

        // 创建乱序的有雾图
        ImageUrlVO hazyHeavy = new ImageUrlVO();
        hazyHeavy.setId(104L);
        hazyHeavy.setType("hazy");
        hazyHeavy.setHazeLevel("heavy");
        imageUrlVOs.add(hazyHeavy);

        ImageUrlVO hazyLight = new ImageUrlVO();
        hazyLight.setId(102L);
        hazyLight.setType("hazy");
        hazyLight.setHazeLevel("light");
        imageUrlVOs.add(hazyLight);

        ImageUrlVO hazyMedium = new ImageUrlVO();
        hazyMedium.setId(103L);
        hazyMedium.setType("hazy");
        hazyMedium.setHazeLevel("medium");
        imageUrlVOs.add(hazyMedium);

        when(sysItemFileService.getImageUrlVOs(itemId)).thenReturn(imageUrlVOs);

        // Act
        DatasetItemVO result = sysDatasetItemService.getDatasetItem(itemId);

        // Assert
        assertNotNull(result);
        assertNotNull(result.getHazyImages());
        assertEquals(3, result.getHazyImages().size());

        // 验证排序：light < medium < heavy
        assertEquals("light", result.getHazyImages().get(0).getHazeLevel());
        assertEquals("medium", result.getHazyImages().get(1).getHazeLevel());
        assertEquals("heavy", result.getHazyImages().get(2).getHazeLevel());
    }

    /**
     * 测试获取数据项详情（有雾图部分为null）
     * 测试场景：有雾图的hazeLevel部分为null
     * 验证内容：
     * 1. null值排在最后
     * 2. 不抛出异常
     */
    @Test
    @DisplayName("getDatasetItem - 有雾图hazeLevel为null的排序")
    void testGetDatasetItem_HazyImageNullHazeLevelSorting() {
        // Arrange
        Long itemId = 1L;

        when(sysDatasetItemMapper.selectById(itemId)).thenReturn(testDatasetItem);
        List<ImageUrlVO> imageUrlVOs = new ArrayList<>();

        ImageUrlVO hazyMedium = new ImageUrlVO();
        hazyMedium.setId(103L);
        hazyMedium.setType("hazy");
        hazyMedium.setHazeLevel("medium");
        imageUrlVOs.add(hazyMedium);

        ImageUrlVO hazyNull = new ImageUrlVO();
        hazyNull.setId(104L);
        hazyNull.setType("hazy");
        hazyNull.setHazeLevel(null);
        imageUrlVOs.add(hazyNull);

        ImageUrlVO hazyLight = new ImageUrlVO();
        hazyLight.setId(102L);
        hazyLight.setType("hazy");
        hazyLight.setHazeLevel("light");
        imageUrlVOs.add(hazyLight);

        when(sysItemFileService.getImageUrlVOs(itemId)).thenReturn(imageUrlVOs);

        // Act
        DatasetItemVO result = sysDatasetItemService.getDatasetItem(itemId);

        // Assert
        assertNotNull(result);
        assertNotNull(result.getHazyImages());
        assertEquals(3, result.getHazyImages().size());

        // null排在最后
        assertEquals("light", result.getHazyImages().get(0).getHazeLevel());
        assertEquals("medium", result.getHazyImages().get(1).getHazeLevel());
        assertNull(result.getHazyImages().get(2).getHazeLevel());
    }

    /**
     * 测试获取不存在的数据项
     * 测试场景：查询不存在的数据项ID
     * 验证内容：
     * 1. 抛出RuntimeException
     * 2. 异常消息正确
     */
    @Test
    @DisplayName("getDatasetItem - 数据项不存在抛出异常")
    void testGetDatasetItem_NotExists() {
        // Arrange
        Long itemId = 999L;

        when(sysDatasetItemMapper.selectById(itemId)).thenReturn(null);

        // Act & Assert
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            sysDatasetItemService.getDatasetItem(itemId);
        });

        assertEquals("数据项不存在", exception.getMessage());
        verify(sysDatasetItemMapper, times(1)).selectById(itemId);
        verify(sysItemFileService, never()).getImageUrlVOs(anyLong());
    }

    // ==================== 分页查询测试 ====================

    /**
     * 测试分页查询（正常分页查询）
     * 测试场景：正常分页查询
     * 验证内容：
     * 1. 返回分页结果
     * 2. 数据项包含完整的图片信息
     */
    @Test
    @DisplayName("pageSearchDatasetItems - 正常分页查询")
    void testPageSearchDatasetItems() {
        // Arrange
        DatasetItemQuery query = new DatasetItemQuery();
        query.setDatasetId(10L);
        query.setPageNum(1);
        query.setPageSize(20);

        List<SysDataset> leafDatasets = new ArrayList<>();
        SysDataset leafDataset = new SysDataset();
        leafDataset.setId(10L);
        leafDataset.setParentId(null);
        leafDatasets.add(leafDataset);

        when(sysDatasetMapper.selectList(null)).thenReturn(leafDatasets);

        List<DatasetItemVO> searchResults = new ArrayList<>();
        DatasetItemVO itemVO = new DatasetItemVO();
        itemVO.setId(1L);
        itemVO.setDatasetId(10L);
        itemVO.setName("测试数据项");
        itemVO.setCreateTime(LocalDateTime.now());
        searchResults.add(itemVO);

        when(sysDatasetItemMapper.searchImages(any(), anyList(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(), any())).thenReturn(searchResults);

        // Mock sysItemFileService.list()返回图片文件
        when(sysItemFileService.list(any(LambdaQueryWrapper.class))).thenReturn(testItemFiles);

        // Mock convertToImageUrlVO方法返回ImageUrlVO
        when(sysItemFileService.convertToImageUrlVO(any(SysItemFile.class))).thenAnswer(invocation -> {
            SysItemFile itemFile = invocation.getArgument(0);
            ImageUrlVO urlVO = new ImageUrlVO();
            urlVO.setId(itemFile.getId());
            urlVO.setUrl("http://test.com/file_" + itemFile.getId() + ".jpg");
            urlVO.setType(itemFile.getType());
            urlVO.setSizeBytes(1000L + itemFile.getId());
            return urlVO;
        });

        // Act
        Page<DatasetItemVO> result = sysDatasetItemService.pageSearchDatasetItems(query);

        // Assert
        assertNotNull(result);
        assertEquals(1, result.getRecords().size());
        assertEquals(1, result.getCurrent());
        assertEquals(20, result.getSize());

        DatasetItemVO resultItem = result.getRecords().get(0);
        assertEquals(1L, resultItem.getId());
        assertEquals(4, resultItem.getImageCount());
        assertNotNull(resultItem.getClearImage());
        assertNotNull(resultItem.getHazyImages());
    }

    /**
     * 测试分页查询（空结果）
     * 测试场景：查询条件没有匹配的数据
     * 验证内容：
     * 1. 返回空的分页结果
     * 2. total为0
     */
    @Test
    @DisplayName("pageSearchDatasetItems - 空结果")
    void testPageSearchDatasetItems_EmptyResult() {
        // Arrange
        DatasetItemQuery query = new DatasetItemQuery();
        query.setDatasetId(10L);
        query.setPageNum(1);
        query.setPageSize(20);

        List<SysDataset> leafDatasets = new ArrayList<>();
        SysDataset leafDataset = new SysDataset();
        leafDataset.setId(10L);
        leafDataset.setParentId(null);
        leafDatasets.add(leafDataset);

        when(sysDatasetMapper.selectList(null)).thenReturn(leafDatasets);
        when(sysDatasetItemMapper.searchImages(any(), anyList(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(), any())).thenReturn(Collections.emptyList());

        // Act
        Page<DatasetItemVO> result = sysDatasetItemService.pageSearchDatasetItems(query);

        // Assert
        assertNotNull(result);
        assertTrue(result.getRecords().isEmpty());
    }

    /**
     * 测试分页查询（无叶子数据集）
     * 测试场景：数据集下没有叶子节点
     * 验证内容：
     * 1. 返回空的分页结果
     * 2. 不会调用searchImages方法
     */
    @Test
    @DisplayName("pageSearchDatasetItems - 无叶子数据集")
    void testPageSearchDatasetItems_NoLeafDatasets() {
        // Arrange
        DatasetItemQuery query = new DatasetItemQuery();
        query.setDatasetId(10L);
        query.setPageNum(1);
        query.setPageSize(20);

        when(sysDatasetMapper.selectList(null)).thenReturn(Collections.emptyList());

        // Act
        Page<DatasetItemVO> result = sysDatasetItemService.pageSearchDatasetItems(query);

        // Assert
        assertNotNull(result);
        assertTrue(result.getRecords().isEmpty());
        assertEquals(0L, result.getTotal());
        verify(sysDatasetItemMapper, never()).searchImages(any(), anyList(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(), any());
    }

    // ==================== 辅助方法 ====================

    /**
     * 创建模拟的ImageUrlVO列表
     */
    private List<ImageUrlVO> createMockImageUrlVOs() {
        List<ImageUrlVO> imageUrlVOs = new ArrayList<>();

        ImageUrlVO clearImage = new ImageUrlVO();
        clearImage.setId(101L);
        clearImage.setType("clear");
        clearImage.setUrl("https://cdn.example.com/clear/001.jpg");
        clearImage.setThumbnailUrl("https://cdn.example.com/clear/thumb_001.jpg");
        clearImage.setWidth(1920);
        clearImage.setHeight(1080);
        clearImage.setSceneType("indoor");
        imageUrlVOs.add(clearImage);

        ImageUrlVO hazyLight = new ImageUrlVO();
        hazyLight.setId(102L);
        hazyLight.setType("hazy");
        hazyLight.setHazeLevel("light");
        hazyLight.setUrl("https://cdn.example.com/hazy/001_light.jpg");
        hazyLight.setThumbnailUrl("https://cdn.example.com/hazy/thumb_001_light.jpg");
        hazyLight.setWidth(1920);
        hazyLight.setHeight(1080);
        hazyLight.setSceneType("outdoor");
        imageUrlVOs.add(hazyLight);

        ImageUrlVO hazyMedium = new ImageUrlVO();
        hazyMedium.setId(103L);
        hazyMedium.setType("hazy");
        hazyMedium.setHazeLevel("medium");
        hazyMedium.setUrl("https://cdn.example.com/hazy/001_medium.jpg");
        hazyMedium.setThumbnailUrl("https://cdn.example.com/hazy/thumb_001_medium.jpg");
        hazyMedium.setWidth(1920);
        hazyMedium.setHeight(1080);
        hazyMedium.setSceneType("outdoor");
        imageUrlVOs.add(hazyMedium);

        ImageUrlVO hazyHeavy = new ImageUrlVO();
        hazyHeavy.setId(104L);
        hazyHeavy.setType("hazy");
        hazyHeavy.setHazeLevel("heavy");
        hazyHeavy.setUrl("https://cdn.example.com/hazy/001_heavy.jpg");
        hazyHeavy.setThumbnailUrl("https://cdn.example.com/hazy/thumb_001_heavy.jpg");
        hazyHeavy.setWidth(1920);
        hazyHeavy.setHeight(1080);
        hazyHeavy.setSceneType("outdoor");
        imageUrlVOs.add(hazyHeavy);

        return imageUrlVOs;
    }
}
