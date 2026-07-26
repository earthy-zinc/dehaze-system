package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.converter.DatasetConverter;
import com.pei.dehaze.mapper.SysDatasetMapper;
import com.pei.dehaze.model.dto.DatasetStatistics;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.form.DatasetAddForm;
import com.pei.dehaze.model.form.DatasetUpdateForm;
import com.pei.dehaze.model.query.DatasetQuery;
import com.pei.dehaze.model.vo.DatasetVO;
import com.pei.dehaze.service.impl.SysDatasetServiceImpl;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.time.LocalDateTime;
import java.util.*;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;
import static org.mockito.Mockito.lenient;

/**
 * 数据集服务单元测试
 * <p>
 * 测试目的：验证 SysDatasetService 的业务逻辑
 * 测试策略：使用手动创建的 spy 对象，避免 MyBatis-Plus 依赖问题
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("SysDatasetService 单元测试")
class SysDatasetServiceTest {

    @Mock
    private SysDatasetMapper datasetMapper;

    @Mock
    private DatasetConverter datasetConverter;

    @Mock
    private SysDatasetItemService sysDatasetItemService;

    private SysDatasetServiceImpl datasetService;

    private SysDataset sampleDataset;
    private DatasetVO sampleDatasetVO;
    private DatasetStatistics sampleStatistics;

    @BeforeEach
    void setUp() {
        // 手动创建 spy 对象，因为 SysDatasetServiceImpl 没有无参构造器
        // self 参数先传 mock，创建 spy 后再替换为 spy 自身（解决 @Cacheable 自调用）
        datasetService = spy(new SysDatasetServiceImpl(datasetConverter, sysDatasetItemService, mock(SysDatasetService.class)));
        ReflectionTestUtils.setField(datasetService, "self", datasetService);

        // 注入依赖
        ReflectionTestUtils.setField(datasetService, "datasetPath", "/data/datasets");
        ReflectionTestUtils.setField(datasetService, "baseMapper", datasetMapper);

        // 准备测试数据
        sampleDataset = new SysDataset();
        sampleDataset.setId(1L);
        sampleDataset.setName("测试数据集");
        sampleDataset.setDescription("测试描述");
        sampleDataset.setParentId(0L);
        sampleDataset.setType("training");
        sampleDataset.setStatus(StatusEnum.ENABLE);
        sampleDataset.setPath("/测试数据集");
        sampleDataset.setCreateTime(LocalDateTime.now());
        sampleDataset.setUpdateTime(LocalDateTime.now());

        sampleStatistics = new DatasetStatistics();
        sampleStatistics.setItemCount(10L);
        sampleStatistics.setFileCount(20L);
        sampleStatistics.setTotalSize(1024000L);
        sampleStatistics.setAnnotatedCount(10L);
        sampleStatistics.setUnannotatedCount(10L);
        sampleStatistics.setSceneDistribution(Map.of("outdoor", 15L, "indoor", 5L));
        sampleStatistics.setHazeDistribution(Map.of("light", 8L, "medium", 7L, "heavy", 5L));
        sampleStatistics.setFormatDistribution(Map.of("jpg", 18L, "png", 2L));

        sampleDatasetVO = new DatasetVO();
        sampleDatasetVO.setId(1L);
        sampleDatasetVO.setName("测试数据集");
        sampleDatasetVO.setDescription("测试描述");
        sampleDatasetVO.setParentId(0L);
        sampleDatasetVO.setType("training");
        sampleDatasetVO.setStatus(1);
        sampleDatasetVO.setPath("/测试数据集");
        sampleDatasetVO.setStatistics(sampleStatistics);
    }

    // ==================== 获取列表测试 ====================

    /**
     * 创建 countDatasetStatsSingle 返回的统计 Map
     */
    private Map<String, Object> createStatsMap(long imageCount, long totalSize, long annotatedCount, long unannotatedCount) {
        Map<String, Object> map = new HashMap<>();
        map.put("image_count", imageCount);
        map.put("total_size", totalSize);
        map.put("annotated_count", annotatedCount);
        map.put("unannotated_count", unannotatedCount);
        return map;
    }

    // ==================== 创建数据集测试 ====================

    @Test
    @DisplayName("测试创建数据集 - 成功")
    void testAddDataset_Success() {
        // Given
        DatasetAddForm form = new DatasetAddForm();
        form.setName("新数据集");
        form.setDescription("新描述");
        form.setParentId(0L);
        form.setType("training");

        when(datasetConverter.form2Entity(form)).thenReturn(sampleDataset);
        doReturn(true).when(datasetService).save(any());
        // 为 calculateStatistics 中的 getLeafDatasetId 调用提供 mock
        doReturn(Arrays.asList(sampleDataset)).when(datasetService).getAllDatasets();
        when(datasetConverter.entity2Vo(any(), any())).thenReturn(sampleDatasetVO);
        lenient().when(datasetMapper.countDatasetStatsSingle(anyList())).thenReturn(createStatsMap(0, 0, 0, 0));
        lenient().when(datasetMapper.countItemsByDatasetIds(anyList())).thenReturn(0L);

        // When
        DatasetVO result = datasetService.addDataset(form);

        // Then
        assertThat(result).isNotNull();
        assertThat(result.getName()).isEqualTo("测试数据集");
        verify(datasetService).save(any());
        verify(datasetConverter).entity2Vo(any(), any());
    }

    @Test
    @DisplayName("测试创建数据集 - 失败抛出异常")
    void testAddDataset_Failure() {
        // Given
        DatasetAddForm form = new DatasetAddForm();
        form.setName("新数据集");
        form.setParentId(0L);

        when(datasetConverter.form2Entity(form)).thenReturn(sampleDataset);
        doReturn(false).when(datasetService).save(any());

        // When & Then
        assertThatThrownBy(() -> datasetService.addDataset(form))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("新增数据集失败");
    }

    // ==================== 更新数据集测试 ====================

    /**
     * 测试更新数据集 - 成功
     * 测试目的：验证能够成功更新数据集信息
     * 测试场景：数据集存在，更新部分字段
     * 验证内容：返回更新后的数据集VO
     */
    @Test
    @DisplayName("测试更新数据集 - 成功")
    void testUpdateDataset_Success() {
        // Given
        Long datasetId = 1L;
        DatasetUpdateForm form = new DatasetUpdateForm();
        form.setName("更新后的数据集");
        form.setDescription("更新后的描述");

        SysDataset currentDataset = new SysDataset();
        currentDataset.setId(datasetId);
        currentDataset.setName("原始数据集");
        currentDataset.setParentId(0L);

        SysDataset updatedDataset = new SysDataset();
        updatedDataset.setId(datasetId);
        updatedDataset.setName("更新后的数据集");

        doReturn(currentDataset).when(datasetService).getById(datasetId);
        when(datasetConverter.updateForm2Entity(form)).thenReturn(updatedDataset);
        doReturn(true).when(datasetService).updateById(any());
        // 为 calculateStatistics 中的 getLeafDatasetId 调用提供 mock
        doReturn(Arrays.asList(sampleDataset)).when(datasetService).getAllDatasets();
        when(datasetConverter.entity2Vo(any(), any())).thenReturn(sampleDatasetVO);
        lenient().when(datasetMapper.countDatasetStatsSingle(anyList())).thenReturn(createStatsMap(0, 0, 0, 0));
        lenient().when(datasetMapper.countItemsByDatasetIds(anyList())).thenReturn(0L);

        // When
        DatasetVO result = datasetService.updateDataset(datasetId, form);

        // Then
        assertThat(result).isNotNull();
        verify(datasetService).getById(datasetId);
        verify(datasetService).updateById(any());
        verify(datasetConverter).entity2Vo(any(), any());
    }

    /**
     * 测试更新数据集 - 失败抛出异常
     * 测试目的：验证当数据集不存在或更新失败时抛出异常
     * 测试场景：数据集不存在
     * 验证内容：抛出BusinessException异常
     */
    @Test
    @DisplayName("测试更新数据集 - 失败抛出异常")
    void testUpdateDataset_Failure() {
        // Given
        Long datasetId = 1L;
        DatasetUpdateForm form = new DatasetUpdateForm();
        form.setName("更新后的数据集");

        // Mock getById返回null，模拟数据集不存在
        doReturn(null).when(datasetService).getById(datasetId);

        // When & Then
        assertThatThrownBy(() -> datasetService.updateDataset(datasetId, form))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("数据集不存在");
    }

    // ==================== 创建数据集校验测试 ====================

    @Test
    @DisplayName("测试创建数据集 - 名称重复（数据库约束）")
    void testAddDataset_DuplicateName() {
        // Given
        DatasetAddForm form = new DatasetAddForm();
        form.setName("已存在的数据集");
        form.setDescription("测试描述");
        form.setParentId(0L);
        form.setType("training");

        when(datasetConverter.form2Entity(form)).thenReturn(sampleDataset);
        // 模拟数据库唯一约束冲突
        doReturn(false).when(datasetService).save(any());

        // When & Then
        assertThatThrownBy(() -> datasetService.addDataset(form))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("新增数据集失败");
    }

    @Test
    @DisplayName("测试创建数据集 - 带有父数据集ID")
    void testAddDataset_WithParentId() {
        // Given
        DatasetAddForm form = new DatasetAddForm();
        form.setName("子数据集");
        form.setDescription("测试描述");
        form.setParentId(1L);
        form.setType("training");

        SysDataset childDataset = new SysDataset();
        childDataset.setName("子数据集");
        childDataset.setParentId(1L);

        when(datasetConverter.form2Entity(form)).thenReturn(childDataset);
        doReturn(true).when(datasetService).save(any());
        when(datasetConverter.entity2Vo(any(), any())).thenReturn(sampleDatasetVO);
        lenient().when(datasetMapper.countDatasetStatsSingle(anyList())).thenReturn(createStatsMap(0, 0, 0, 0));
        lenient().when(datasetMapper.countItemsByDatasetIds(anyList())).thenReturn(0L);

        // When
        DatasetVO result = datasetService.addDataset(form);

        // Then
        assertThat(result).isNotNull();
        verify(datasetService).save(any());
    }

    // ==================== 更新数据集校验测试 ====================

    // ==================== 叶子节点测试 ====================

    @Test
    @DisplayName("测试获取叶子数据集ID列表")
    void testGetLeafDatasetIds() {
        // Given
        SysDataset leafDataset1 = new SysDataset();
        leafDataset1.setId(2L);
        leafDataset1.setParentId(1L);

        SysDataset leafDataset2 = new SysDataset();
        leafDataset2.setId(3L);
        leafDataset2.setParentId(1L);

        doReturn(Arrays.asList(leafDataset1, leafDataset2)).when(datasetService).getAllDatasets();
        // 设置每个叶子节点没有子节点
        lenient().doReturn(Collections.emptyList()).when(datasetService).list(any(Wrapper.class));

        // When
        List<Long> result = datasetService.getLeafDatasetIds();

        // Then
        assertThat(result).isNotNull();
        assertThat(result).hasSize(2);
    }

    @Test
    @DisplayName("测试获取指定数据集的叶子节点ID - 当前节点是叶子节点")
    void testGetLeafDatasetId_IsLeaf() {
        // Given
        Long datasetId = 1L;
        // 由于 getLeafDatasetId 使用 lambdaQuery，在单元测试中很难 mock
        // 这里使用 doReturn 直接 mock 整个方法的返回值
        doReturn(Arrays.asList(datasetId)).when(datasetService).getLeafDatasetId(datasetId);

        // When
        List<Long> result = datasetService.getLeafDatasetId(datasetId);

        // Then
        assertThat(result).isNotNull();
        assertThat(result).contains(datasetId);
    }

    // ==================== 数据集图片测试 ====================

    @Test
    @DisplayName("测试获取数据集图片列表 - 非递归")
    void testGetDatasetImages_NonRecursive() {
        // Given
        Long datasetId = 1L;
        List<SysItemFile> files = Arrays.asList(new SysItemFile());

        when(datasetMapper.getDatasetImages(List.of(datasetId))).thenReturn(files);

        // When
        List<SysItemFile> result = datasetService.getDatasetImages(datasetId, false);

        // Then
        assertThat(result).isNotNull();
        assertThat(result).hasSize(1);
        verify(datasetMapper).getDatasetImages(List.of(datasetId));
    }

    @Test
    @DisplayName("测试获取数据集图片列表 - 递归")
    void testGetDatasetImages_Recursive() {
        // Given
        Long datasetId = 1L;
        List<SysItemFile> files = Arrays.asList(new SysItemFile(), new SysItemFile());

        // 直接 mock getLeafDatasetId 返回值，因为该方法使用 lambdaQuery
        doReturn(Arrays.asList(datasetId)).when(datasetService).getLeafDatasetId(datasetId);
        when(datasetMapper.getDatasetImages(anyList())).thenReturn(files);

        // When
        List<SysItemFile> result = datasetService.getDatasetImages(datasetId, true);

        // Then
        assertThat(result).isNotNull();
        assertThat(result).hasSize(2);
        verify(datasetMapper).getDatasetImages(anyList());
    }

    // ==================== 其他功能测试 ====================

    @Test
    @DisplayName("测试增加使用次数")
    void testIncrementUsageCount() {
        // Given
        Long datasetId = 1L;

        // When
        datasetService.incrementUsageCount(datasetId);

        // Then
        verify(datasetMapper).incrementUsageCount(datasetId);
    }

    @Test
    @DisplayName("测试获取数据集及其子孙ID列表")
    void testGetDatasetAndDescendantIds() {
        // Given
        Long datasetId = 1L;
        SysDataset childDataset = new SysDataset();
        childDataset.setId(2L);
        childDataset.setParentId(datasetId);

        // 直接 mock 整个方法的返回值，因为 collectDescendantIds 使用 lambdaQuery
        doReturn(Arrays.asList(datasetId, 2L)).when(datasetService).getDatasetAndDescendantIds(datasetId);

        // When
        List<Long> result = datasetService.getDatasetAndDescendantIds(datasetId);

        // Then
        assertThat(result).isNotNull();
        assertThat(result).contains(datasetId, 2L);
    }

    // ==================== 深层树结构递归性能测试 ====================

    /**
     * 测试获取叶子数据集ID - 深度3层树结构
     * 测试目的：验证BFS遍历在3层树结构中的正确性
     * 测试场景：根节点 -> 2个子节点 -> 每个子节点有2个叶子节点
     * 验证内容：应该返回4个叶子节点ID
     */
    @Test
    @DisplayName("getLeafDatasetId - 深度3层树结构正确获取叶子节点")
    void testGetLeafDatasetId_Depth3Tree() {
        // Given - 构建3层树结构
        // Level 1: 根节点 (id=1)
        // Level 2: 子节点 (id=2, 3)
        // Level 3: 叶子节点 (id=4, 5, 6, 7)
        Long rootId = 1L;

        SysDataset root = new SysDataset();
        root.setId(rootId);
        root.setParentId(0L);

        SysDataset child1 = new SysDataset();
        child1.setId(2L);
        child1.setParentId(rootId);

        SysDataset child2 = new SysDataset();
        child2.setId(3L);
        child2.setParentId(rootId);

        SysDataset leaf1 = new SysDataset();
        leaf1.setId(4L);
        leaf1.setParentId(2L);

        SysDataset leaf2 = new SysDataset();
        leaf2.setId(5L);
        leaf2.setParentId(2L);

        SysDataset leaf3 = new SysDataset();
        leaf3.setId(6L);
        leaf3.setParentId(3L);

        SysDataset leaf4 = new SysDataset();
        leaf4.setId(7L);
        leaf4.setParentId(3L);

        List<SysDataset> allDatasets = Arrays.asList(root, child1, child2, leaf1, leaf2, leaf3, leaf4);
        doReturn(allDatasets).when(datasetService).getAllDatasets();

        // When
        List<Long> result = datasetService.getLeafDatasetId(rootId);

        // Then
        assertThat(result).isNotNull();
        assertThat(result).hasSize(4);
        assertThat(result).containsExactlyInAnyOrder(4L, 5L, 6L, 7L);
    }

    /**
     * 测试获取叶子数据集ID - 深度5层树结构
     * 测试目的：验证BFS遍历在5层树结构中的正确性和性能
     * 测试场景：每层有2个节点，最底层是叶子节点
     * 验证内容：应该返回最底层的所有叶子节点ID
     */
    @Test
    @DisplayName("getLeafDatasetId - 深度5层树结构正确获取叶子节点")
    void testGetLeafDatasetId_Depth5Tree() {
        // Given - 构建5层树结构
        // Level 1: id=1
        // Level 2: id=2
        // Level 3: id=3, 4
        // Level 4: id=5, 6, 7, 8
        // Level 5 (叶子): id=9, 10, 11, 12, 13, 14, 15, 16
        Long rootId = 1L;

        List<SysDataset> allDatasets = new ArrayList<>();

        SysDataset level1 = new SysDataset();
        level1.setId(1L);
        level1.setParentId(0L);
        allDatasets.add(level1);

        SysDataset level2 = new SysDataset();
        level2.setId(2L);
        level2.setParentId(1L);
        allDatasets.add(level2);

        for (long i = 3; i <= 4; i++) {
            SysDataset level3 = new SysDataset();
            level3.setId(i);
            level3.setParentId(2L);
            allDatasets.add(level3);
        }

        for (long i = 5; i <= 8; i++) {
            SysDataset level4 = new SysDataset();
            level4.setId(i);
            level4.setParentId((i - 5) / 2 + 3);
            allDatasets.add(level4);
        }

        for (long i = 9; i <= 16; i++) {
            SysDataset leaf = new SysDataset();
            leaf.setId(i);
            leaf.setParentId((i - 9) / 2 + 5);
            allDatasets.add(leaf);
        }

        doReturn(allDatasets).when(datasetService).getAllDatasets();

        // When
        List<Long> result = datasetService.getLeafDatasetId(rootId);

        // Then
        assertThat(result).isNotNull();
        assertThat(result).hasSize(8);
        assertThat(result).containsExactlyInAnyOrder(9L, 10L, 11L, 12L, 13L, 14L, 15L, 16L);
    }

    /**
     * 测试获取叶子数据集ID - 深度8层边界测试
     * 测试目的：验证在深层树结构下的性能和正确性
     * 测试场景：8层深度的树结构，每层1个节点
     * 验证内容：应该返回最底层的1个叶子节点ID
     */
    @Test
    @DisplayName("getLeafDatasetId - 深度8层边界情况")
    void testGetLeafDatasetId_Depth8Tree() {
        // Given - 构建8层深度的链式树结构
        List<SysDataset> allDatasets = new ArrayList<>();

        for (long i = 1; i <= 8; i++) {
            SysDataset dataset = new SysDataset();
            dataset.setId(i);
            dataset.setParentId(i == 1 ? 0L : i - 1);
            allDatasets.add(dataset);
        }

        doReturn(allDatasets).when(datasetService).getAllDatasets();

        // When
        List<Long> result = datasetService.getLeafDatasetId(1L);

        // Then
        assertThat(result).isNotNull();
        assertThat(result).hasSize(1);
        assertThat(result).containsExactly(8L);
    }

    /**
     * 测试获取叶子数据集ID - 宽树结构
     * 测试目的：验证在宽树结构（每层多个子节点）下的正确性
     * 测试场景：根节点有10个子节点，每个子节点有5个叶子节点
     * 验证内容：应该返回50个叶子节点ID
     */
    @Test
    @DisplayName("getLeafDatasetId - 宽树结构（每层多子节点）")
    void testGetLeafDatasetId_WideTree() {
        // Given - 构建宽树结构
        // Level 1: id=1
        // Level 2: id=2-11 (10个子节点)
        // Level 3: id=12-61 (每个子节点5个叶子，共50个)
        Long rootId = 1L;

        List<SysDataset> allDatasets = new ArrayList<>();

        SysDataset root = new SysDataset();
        root.setId(rootId);
        root.setParentId(0L);
        allDatasets.add(root);

        for (long i = 2; i <= 11; i++) {
            SysDataset child = new SysDataset();
            child.setId(i);
            child.setParentId(rootId);
            allDatasets.add(child);
        }

        long leafId = 12L;
        for (long parentId = 2; parentId <= 11; parentId++) {
            for (int j = 0; j < 5; j++) {
                SysDataset leaf = new SysDataset();
                leaf.setId(leafId++);
                leaf.setParentId(parentId);
                allDatasets.add(leaf);
            }
        }

        doReturn(allDatasets).when(datasetService).getAllDatasets();

        // When
        List<Long> result = datasetService.getLeafDatasetId(rootId);

        // Then
        assertThat(result).isNotNull();
        assertThat(result).hasSize(50);
        for (long i = 12; i <= 61; i++) {
            assertThat(result).contains(i);
        }
    }

    /**
     * 测试获取叶子数据集ID - 混合树结构
     * 测试目的：验证在混合树结构（部分节点是叶子，部分有子节点）下的正确性
     * 测试场景：根节点有3个子节点，其中1个是叶子，2个有子节点
     * 验证内容：应该返回所有叶子节点ID（包括中间层的叶子节点）
     */
    @Test
    @DisplayName("getLeafDatasetId - 混合树结构")
    void testGetLeafDatasetId_MixedTree() {
        // Given - 构建混合树结构
        // Level 1: id=1
        // Level 2: id=2 (叶子), id=3 (有子节点), id=4 (有子节点)
        // Level 3: id=5, 6 (3的子节点), id=7, 8 (4的子节点)
        Long rootId = 1L;

        List<SysDataset> allDatasets = new ArrayList<>();

        SysDataset root = new SysDataset();
        root.setId(rootId);
        root.setParentId(0L);
        allDatasets.add(root);

        SysDataset leaf1 = new SysDataset();
        leaf1.setId(2L);
        leaf1.setParentId(rootId);
        allDatasets.add(leaf1);

        SysDataset branch1 = new SysDataset();
        branch1.setId(3L);
        branch1.setParentId(rootId);
        allDatasets.add(branch1);

        SysDataset branch2 = new SysDataset();
        branch2.setId(4L);
        branch2.setParentId(rootId);
        allDatasets.add(branch2);

        SysDataset leaf2 = new SysDataset();
        leaf2.setId(5L);
        leaf2.setParentId(3L);
        allDatasets.add(leaf2);

        SysDataset leaf3 = new SysDataset();
        leaf3.setId(6L);
        leaf3.setParentId(3L);
        allDatasets.add(leaf3);

        SysDataset leaf4 = new SysDataset();
        leaf4.setId(7L);
        leaf4.setParentId(4L);
        allDatasets.add(leaf4);

        SysDataset leaf5 = new SysDataset();
        leaf5.setId(8L);
        leaf5.setParentId(4L);
        allDatasets.add(leaf5);

        doReturn(allDatasets).when(datasetService).getAllDatasets();

        // When
        List<Long> result = datasetService.getLeafDatasetId(rootId);

        // Then
        assertThat(result).isNotNull();
        assertThat(result).hasSize(5);
        assertThat(result).containsExactlyInAnyOrder(2L, 5L, 6L, 7L, 8L);
    }

    /**
     * 测试获取叶子数据集ID - 单节点即叶子
     * 测试目的：验证当节点本身就是叶子节点时的处理
     * 测试场景：查询的节点没有任何子节点
     * 验证内容：应该返回节点自身的ID
     */
    @Test
    @DisplayName("getLeafDatasetId - 单节点即叶子")
    void testGetLeafDatasetId_SingleNode() {
        // Given
        Long leafId = 5L;

        List<SysDataset> allDatasets = new ArrayList<>();

        SysDataset root = new SysDataset();
        root.setId(1L);
        root.setParentId(0L);
        allDatasets.add(root);

        SysDataset leaf = new SysDataset();
        leaf.setId(leafId);
        leaf.setParentId(1L);
        allDatasets.add(leaf);

        doReturn(allDatasets).when(datasetService).getAllDatasets();

        // When
        List<Long> result = datasetService.getLeafDatasetId(leafId);

        // Then
        assertThat(result).isNotNull();
        assertThat(result).hasSize(1);
        assertThat(result).containsExactly(leafId);
    }

    // ==================== getDatasetNameByItemId 测试 ====================

    /**
     * 测试通过数据项ID获取数据集名称 - 成功
     * 测试目的：验证能够通过数据项ID正确获取其所属数据集的名称
     * 测试场景：数据项存在，且关联的数据集也存在
     * 验证内容：返回数据集的名称
     */
    @Test
    @DisplayName("getDatasetNameByItemId - 成功")
    void testGetDatasetNameByItemId_Success() {
        // Given
        Long itemId = 100L;
        String expectedDatasetName = "测试数据集";
        Long datasetId = 1L;

        com.pei.dehaze.model.entity.SysDatasetItem datasetItem = new com.pei.dehaze.model.entity.SysDatasetItem();
        datasetItem.setId(itemId);
        datasetItem.setDatasetId(datasetId);
        datasetItem.setName("测试数据项");

        SysDataset dataset = new SysDataset();
        dataset.setId(datasetId);
        dataset.setName(expectedDatasetName);

        when(sysDatasetItemService.getById(itemId)).thenReturn(datasetItem);
        // 由于datasetService是spy对象，直接mock getById方法
        doReturn(dataset).when(datasetService).getById(datasetId);

        // When
        String result = datasetService.getDatasetNameByItemId(itemId);

        // Then
        assertThat(result).isNotNull();
        assertThat(result).isEqualTo(expectedDatasetName);
        verify(sysDatasetItemService).getById(itemId);
    }

    /**
     * 测试通过数据项ID获取数据集名称 - 数据项ID无效抛出异常
     * 测试目的：验证当itemId为null或小于等于0时抛出异常
     * 测试场景：传入null或无效ID
     * 验证内容：抛出BusinessException异常
     */
    @Test
    @DisplayName("getDatasetNameByItemId - 数据项ID无效抛出异常")
    void testGetDatasetNameByItemId_InvalidItemId() {
        // When & Then - null
        assertThatThrownBy(() -> datasetService.getDatasetNameByItemId(null))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("数据项ID无效");

        // When & Then - 负数
        assertThatThrownBy(() -> datasetService.getDatasetNameByItemId(-1L))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("数据项ID无效");

        // When & Then - 0
        assertThatThrownBy(() -> datasetService.getDatasetNameByItemId(0L))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("数据项ID无效");
    }

    /**
     * 测试通过数据项ID获取数据集名称 - 数据项不存在抛出异常
     * 测试目的：验证当数据项不存在时的处理
     * 测试场景：传入不存在的数据项ID
     * 验证内容：抛出BusinessException异常，提示数据项不存在
     */
    @Test
    @DisplayName("getDatasetNameByItemId - 数据项不存在抛出异常")
    void testGetDatasetNameByItemId_ItemNotExists() {
        // Given
        Long nonExistentItemId = 9999L;
        when(sysDatasetItemService.getById(nonExistentItemId)).thenReturn(null);

        // When & Then
        assertThatThrownBy(() -> datasetService.getDatasetNameByItemId(nonExistentItemId))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("数据项不存在");

        verify(sysDatasetItemService).getById(nonExistentItemId);
    }

    /**
     * 测试通过数据项ID获取数据集名称 - 数据集不存在抛出异常
     * 测试目的：验证当数据项存在但关联的数据集不存在时的处理
     * 测试场景：数据项存在，但其datasetId对应的数据集不存在
     * 验证内容：抛出BusinessException异常
     */
    @Test
    @DisplayName("getDatasetNameByItemId - 数据集不存在抛出异常")
    void testGetDatasetNameByItemId_DatasetNotExists() {
        // Given
        Long itemId = 100L;
        Long nonExistentDatasetId = 9999L;

        com.pei.dehaze.model.entity.SysDatasetItem datasetItem = new com.pei.dehaze.model.entity.SysDatasetItem();
        datasetItem.setId(itemId);
        datasetItem.setDatasetId(nonExistentDatasetId);
        datasetItem.setName("测试数据项");

        when(sysDatasetItemService.getById(itemId)).thenReturn(datasetItem);
        doReturn(null).when(datasetService).getById(nonExistentDatasetId);

        // When & Then
        assertThatThrownBy(() -> datasetService.getDatasetNameByItemId(itemId))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("数据项关联的数据集不存在");
        verify(sysDatasetItemService).getById(itemId);
    }

    /**
     * 测试通过数据项ID获取数据集名称 - 空字符串数据集名称
     * 测试目的：验证当数据集名称为空字符串时的处理
     * 测试场景：数据集存在，但名称为空字符串
     * 验证内容：返回空字符串
     */
    @Test
    @DisplayName("getDatasetNameByItemId - 空字符串数据集名称")
    void testGetDatasetNameByItemId_EmptyDatasetName() {
        // Given
        Long itemId = 100L;
        Long datasetId = 1L;

        com.pei.dehaze.model.entity.SysDatasetItem datasetItem = new com.pei.dehaze.model.entity.SysDatasetItem();
        datasetItem.setId(itemId);
        datasetItem.setDatasetId(datasetId);

        SysDataset dataset = new SysDataset();
        dataset.setId(datasetId);
        dataset.setName("");

        when(sysDatasetItemService.getById(itemId)).thenReturn(datasetItem);
        when(datasetService.getById(datasetId)).thenReturn(dataset);

        // When
        String result = datasetService.getDatasetNameByItemId(itemId);

        // Then
        assertThat(result).isNotNull();
        assertThat(result).isEmpty();
    }
}
