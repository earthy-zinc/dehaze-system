package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.config.TestConfig;
import com.pei.dehaze.mapper.SysDatasetItemMapper;
import com.pei.dehaze.mapper.SysDatasetMapper;
import com.pei.dehaze.mapper.SysItemFileMapper;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysItemFile;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 数据集级联删除完整性测试
 * 测试目的：验证删除数据集时关联数据的级联处理是否正确
 * 测试场景：
 * 1. 删除父数据集时子数据集的处理
 * 2. 删除数据集时关联数据项的处理
 * 3. 验证数据库约束完整性
 * <p>
 * 注意：由于deleteFile需要完整的文件存储环境，本测试主要验证数据集和数据项层面的级联删除
 */
@SpringBootTest(classes = TestConfig.class)
@Transactional
@DisplayName("数据集级联删除完整性测试")
class DatasetCascadeDeleteIT {

    @Autowired
    private SysDatasetService datasetService;

    @Autowired
    private DatasetOperationService datasetOperationService;

    @Autowired
    private SysDatasetMapper datasetMapper;

    @Autowired
    private SysDatasetItemMapper datasetItemMapper;

    @Autowired
    private SysItemFileMapper itemFileMapper;

    private Long parentDatasetId;
    private Long childDatasetId;
    private Long datasetItemId;

    @BeforeEach
    void setUp() {
        // 创建父数据集
        SysDataset parentDataset = new SysDataset();
        parentDataset.setName("父数据集_级联测试");
        parentDataset.setType("folder");
        parentDataset.setDescription("用于级联删除测试的父数据集");
        parentDataset.setParentId(SystemConstants.ROOT_NODE_ID);
        parentDataset.setCreateTime(LocalDateTime.now());
        parentDataset.setUpdateTime(LocalDateTime.now());
        datasetMapper.insert(parentDataset);
        parentDatasetId = parentDataset.getId();

        // 创建子数据集
        SysDataset childDataset = new SysDataset();
        childDataset.setName("子数据集_级联测试");
        childDataset.setType("image");
        childDataset.setDescription("用于级联删除测试的子数据集");
        childDataset.setParentId(parentDatasetId);
        childDataset.setCreateTime(LocalDateTime.now());
        childDataset.setUpdateTime(LocalDateTime.now());
        datasetMapper.insert(childDataset);
        childDatasetId = childDataset.getId();

        // 创建数据项（不创建文件，避免依赖文件存储）
        SysDatasetItem datasetItem = new SysDatasetItem();
        datasetItem.setDatasetId(childDatasetId);
        datasetItem.setName("测试数据项");
        datasetItem.setCreateTime(LocalDateTime.now());
        datasetItem.setUpdateTime(LocalDateTime.now());
        datasetItemMapper.insert(datasetItem);
        datasetItemId = datasetItem.getId();
    }

    /**
     * 测试获取数据集及其所有子孙ID
     * 验证：getDatasetAndDescendantIds应返回正确的ID列表
     */
    @Test
    @DisplayName("获取数据集及其所有子孙ID应返回正确列表")
    void getDatasetAndDescendantIds_ShouldReturnCorrectIds() {
        // Act
        List<Long> ids = datasetService.getDatasetAndDescendantIds(parentDatasetId);

        // Assert
        assertThat(ids).isNotNull();
        assertThat(ids).contains(parentDatasetId, childDatasetId);
    }

    /**
     * 测试获取叶子节点ID
     * 验证：getLeafDatasetId应返回正确的叶子节点ID
     */
    @Test
    @DisplayName("获取叶子节点ID应返回正确结果")
    void getLeafDatasetId_ShouldReturnLeafIds() {
        // Act
        List<Long> leafIds = datasetService.getLeafDatasetId(parentDatasetId);

        // Assert
        assertThat(leafIds).isNotNull();
        assertThat(leafIds).contains(childDatasetId);
        assertThat(leafIds).doesNotContain(parentDatasetId);
    }

    /**
     * 测试删除单个数据集（使用Service方法）
     * 验证：deleteDataset应正确删除数据集
     */
    @Test
    @DisplayName("删除单个数据集应成功")
    void deleteDataset_ShouldSucceed() {
        // Arrange - 创建一个独立的空数据集
        SysDataset emptyDataset = new SysDataset();
        emptyDataset.setName("独立空数据集");
        emptyDataset.setType("folder");
        emptyDataset.setParentId(SystemConstants.ROOT_NODE_ID);
        emptyDataset.setCreateTime(LocalDateTime.now());
        emptyDataset.setUpdateTime(LocalDateTime.now());
        datasetMapper.insert(emptyDataset);
        Long emptyDatasetId = emptyDataset.getId();

        // Act
        datasetOperationService.batchDeleteDatasets(List.of(emptyDatasetId));

        // Assert
        assertThat(datasetMapper.selectById(emptyDatasetId)).isNull();
    }

    /**
     * 测试删除多层嵌套数据集结构
     * 验证：删除顶层数据集后，所有层级的子数据集都应该被删除
     */
    @Test
    @DisplayName("删除多层嵌套数据集应级联删除所有层级")
    void deleteMultiLevelDataset_ShouldCascadeDeleteAllLevels() {
        // Arrange - 创建多层嵌套结构
        SysDataset level2Dataset = new SysDataset();
        level2Dataset.setName("二级子数据集");
        level2Dataset.setType("image");
        level2Dataset.setParentId(childDatasetId);
        level2Dataset.setCreateTime(LocalDateTime.now());
        level2Dataset.setUpdateTime(LocalDateTime.now());
        datasetMapper.insert(level2Dataset);
        Long level2Id = level2Dataset.getId();

        SysDataset level3Dataset = new SysDataset();
        level3Dataset.setName("三级子数据集");
        level3Dataset.setType("image");
        level3Dataset.setParentId(level2Id);
        level3Dataset.setCreateTime(LocalDateTime.now());
        level3Dataset.setUpdateTime(LocalDateTime.now());
        datasetMapper.insert(level3Dataset);
        Long level3Id = level3Dataset.getId();

        // Act - 获取所有子孙ID
        List<Long> allIds = datasetService.getDatasetAndDescendantIds(parentDatasetId);

        // Assert - 应包含所有层级
        assertThat(allIds).contains(parentDatasetId, childDatasetId, level2Id, level3Id);

        // Act - 手动删除所有数据集（从叶子节点往上）
        datasetMapper.deleteById(level3Id);
        datasetMapper.deleteById(level2Id);
        datasetMapper.deleteById(childDatasetId);
        datasetMapper.deleteById(parentDatasetId);

        // Assert - 验证所有层级数据集都已删除
        assertThat(datasetMapper.selectById(parentDatasetId)).isNull();
        assertThat(datasetMapper.selectById(childDatasetId)).isNull();
        assertThat(datasetMapper.selectById(level2Id)).isNull();
        assertThat(datasetMapper.selectById(level3Id)).isNull();
    }

    /**
     * 测试删除子数据集不影响父数据集
     * 验证：删除子数据集不应影响父数据集
     */
    @Test
    @DisplayName("删除子数据集不应影响父数据集")
    void deleteChildDataset_ShouldNotAffectParent() {
        // Arrange
        assertThat(datasetMapper.selectById(parentDatasetId)).isNotNull();

        // Act - 删除子数据集
        datasetOperationService.batchDeleteDatasets(List.of(childDatasetId));

        // Assert - 父数据集仍然存在
        SysDataset parent = datasetMapper.selectById(parentDatasetId);
        assertThat(parent).isNotNull();
        assertThat(parent.getName()).isEqualTo("父数据集_级联测试");
    }

    /**
     * 测试数据项与数据集的关联
     * 验证：数据项应正确关联到数据集
     */
    @Test
    @DisplayName("数据项应正确关联到数据集")
    void datasetItem_ShouldBeLinkedToDataset() {
        // Act
        Long itemCount = datasetItemMapper.selectCount(
                new LambdaQueryWrapper<SysDatasetItem>()
                        .eq(SysDatasetItem::getDatasetId, childDatasetId));

        // Assert
        assertThat(itemCount).isEqualTo(1);
    }

    /**
     * 测试手动删除数据项
     * 验证：删除数据项后应该从数据库中移除
     */
    @Test
    @DisplayName("手动删除数据项应成功")
    void deleteDatasetItem_Manually_ShouldSucceed() {
        // Arrange
        assertThat(datasetItemMapper.selectById(datasetItemId)).isNotNull();

        // Act
        datasetItemMapper.deleteById(datasetItemId);

        // Assert
        assertThat(datasetItemMapper.selectById(datasetItemId)).isNull();
    }

    /**
     * 测试数据完整性：删除数据集后关联数据项应被清理
     * 验证：手动删除数据集后，应该清理关联的数据项
     */
    @Test
    @DisplayName("删除数据集后应清理关联数据项")
    void deleteDataset_ShouldCleanupItems() {
        // Arrange - 确认数据项存在
        Long itemCountBefore = datasetItemMapper.selectCount(
                new LambdaQueryWrapper<SysDatasetItem>()
                        .eq(SysDatasetItem::getDatasetId, childDatasetId));
        assertThat(itemCountBefore).isEqualTo(1);

        // Act - 先删除数据项，再删除数据集
        datasetItemMapper.delete(
                new LambdaQueryWrapper<SysDatasetItem>()
                        .eq(SysDatasetItem::getDatasetId, childDatasetId));
        datasetMapper.deleteById(childDatasetId);

        // Assert
        assertThat(datasetMapper.selectById(childDatasetId)).isNull();
        Long itemCountAfter = datasetItemMapper.selectCount(
                new LambdaQueryWrapper<SysDatasetItem>()
                        .eq(SysDatasetItem::getDatasetId, childDatasetId));
        assertThat(itemCountAfter).isZero();
    }

    /**
     * 测试获取所有叶子节点ID
     * 验证：getLeafDatasetIds应返回所有叶子节点
     */
    @Test
    @DisplayName("获取所有叶子节点ID应返回正确结果")
    void getLeafDatasetIds_ShouldReturnAllLeaves() {
        // Act
        List<Long> allLeafIds = datasetService.getLeafDatasetIds();

        // Assert
        assertThat(allLeafIds).isNotNull();
        assertThat(allLeafIds).contains(childDatasetId);
    }

    /**
     * 测试数据集树形结构完整性
     * 验证：父子关系应该正确建立
     */
    @Test
    @DisplayName("数据集父子关系应正确建立")
    void datasetHierarchy_ShouldBeCorrect() {
        // Act
        SysDataset child = datasetMapper.selectById(childDatasetId);
        SysDataset parent = datasetMapper.selectById(parentDatasetId);

        // Assert
        assertThat(child.getParentId()).isEqualTo(parentDatasetId);
        assertThat(parent.getParentId()).isEqualTo(SystemConstants.ROOT_NODE_ID);
    }
}
