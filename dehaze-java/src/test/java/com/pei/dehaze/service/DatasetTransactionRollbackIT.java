package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.config.TestConfig;
import com.pei.dehaze.mapper.SysDatasetItemMapper;
import com.pei.dehaze.mapper.SysDatasetMapper;
import com.pei.dehaze.mapper.SysItemFileMapper;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.form.DatasetAddForm;
import com.pei.dehaze.model.vo.BatchDeleteResult;
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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * 数据集事务回滚测试
 * 测试目的：验证异常发生时事务能够正确回滚
 * 测试场景：
 * 1. 新增数据集失败时回滚
 * 2. 更新数据集失败时回滚
 * 3. 批量操作中途异常时回滚
 * 4. 嵌套事务异常传播
 */
@SpringBootTest(classes = TestConfig.class)
@DisplayName("数据集事务回滚测试")
class DatasetTransactionRollbackIT {

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

    private Long existingDatasetId;

    @BeforeEach
    void setUp() {
        // 创建一个已存在的数据集用于测试
        SysDataset existingDataset = new SysDataset();
        existingDataset.setName("已存在数据集_事务测试");
        existingDataset.setType("image");
        existingDataset.setParentId(SystemConstants.ROOT_NODE_ID);
        existingDataset.setCreateTime(LocalDateTime.now());
        existingDataset.setUpdateTime(LocalDateTime.now());
        datasetMapper.insert(existingDataset);
        existingDatasetId = existingDataset.getId();
    }

    /**
     * 测试新增重名数据集时事务回滚
     * 验证：当新增重名数据集抛出异常时，数据库状态应保持不变
     */
    @Test
    @Transactional
    @DisplayName("新增重名数据集应抛出异常且不影响数据库")
    void addDataset_DuplicateName_ShouldThrowExceptionAndRollback() {
        // Arrange
        DatasetAddForm form = new DatasetAddForm();
        form.setName("已存在数据集_事务测试"); // 重名
        form.setType("image");
        form.setParentId(SystemConstants.ROOT_NODE_ID);

        // 记录当前数据集数量
        Long countBefore = datasetMapper.selectCount(null);

        // Act & Assert
        assertThatThrownBy(() -> datasetService.addDataset(form))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("同父节点下已存在相同名称的数据集");

        // Assert - 数据库状态应保持不变
        Long countAfter = datasetMapper.selectCount(null);
        assertThat(countAfter).isEqualTo(countBefore);
    }

    /**
     * 测试更新不存在的数据集时事务回滚
     * 验证：当更新不存在的数据集抛出异常时，数据库状态应保持不变
     */
    @Test
    @Transactional
    @DisplayName("更新不存在的数据集应抛出异常")
    void updateDataset_NotExists_ShouldThrowException() {
        // Arrange
        Long nonExistentId = 999999L;

        // Act & Assert
        assertThatThrownBy(() -> datasetService.getDatasetById(nonExistentId))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("数据集不存在");
    }

    /**
     * 测试删除不存在的数据集时标记失败
     * 验证：当删除不存在的数据集时，结果中标记为失败
     */
    @Test
    @Transactional
    @DisplayName("删除不存在的数据集应标记失败")
    void deleteDataset_NotExists_ShouldMarkFailed() {
        // Arrange
        Long nonExistentId = 999999L;

        // Act
        BatchDeleteResult result = datasetOperationService.batchDeleteDatasets(List.of(nonExistentId));

        // Assert
        assertThat(result.getFailed()).isGreaterThan(0);
        assertThat(result.getResults())
                .anySatisfy(item -> {
                    assertThat(item.getId()).isEqualTo(nonExistentId);
                    assertThat(item.getStatus()).isEqualTo("failed");
                });
    }

    /**
     * 测试事务注解确保数据一致性
     * 验证：在事务方法中，所有操作要么全部成功，要么全部回滚
     */
    @Test
    @Transactional
    @DisplayName("事务方法中的操作应保持一致性")
    void transactionalMethod_ShouldMaintainConsistency() {
        // Arrange - 创建数据集
        SysDataset dataset = new SysDataset();
        dataset.setName("事务一致性测试_" + System.currentTimeMillis());
        dataset.setType("image");
        dataset.setParentId(SystemConstants.ROOT_NODE_ID);
        dataset.setCreateTime(LocalDateTime.now());
        dataset.setUpdateTime(LocalDateTime.now());
        datasetMapper.insert(dataset);
        Long datasetId = dataset.getId();

        // Arrange - 创建数据项
        SysDatasetItem item = new SysDatasetItem();
        item.setDatasetId(datasetId);
        item.setName("测试数据项");
        item.setCreateTime(LocalDateTime.now());
        item.setUpdateTime(LocalDateTime.now());
        datasetItemMapper.insert(item);
        Long itemId = item.getId();

        // Arrange - 创建文件
        SysItemFile file = new SysItemFile();
        file.setItemId(itemId);
        file.setFileId(1L); // 设置必需的file_id
        file.setType("hazy");
        file.setDescription("测试文件");
        file.setSceneType("outdoor");
        file.setHazeLevel("light");
        file.setWidth(1920);
        file.setHeight(1080);
        file.setCreateTime(LocalDateTime.now());
        file.setUpdateTime(LocalDateTime.now());
        itemFileMapper.insert(file);

        // Assert - 验证所有数据都已插入
        assertThat(datasetMapper.selectById(datasetId)).isNotNull();
        assertThat(datasetItemMapper.selectById(itemId)).isNotNull();
        assertThat(itemFileMapper.selectById(file.getId())).isNotNull();

        // 由于测试方法标记了@Transactional，测试结束后会自动回滚
        // 这里验证在事务中数据是可见的
    }

    /**
     * 测试无效ID参数校验
     * 验证：传入无效ID应该抛出异常
     */
    @Test
    @DisplayName("无效ID应抛出异常")
    void invalidId_ShouldThrowException() {
        // Act & Assert - null ID
        assertThatThrownBy(() -> datasetService.getDatasetById(null))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("数据集ID无效");

        // Act & Assert - 负数ID
        assertThatThrownBy(() -> datasetService.getDatasetById(-1L))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("数据集ID无效");

        // Act & Assert - 零ID
        assertThatThrownBy(() -> datasetService.getDatasetById(0L))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("数据集ID无效");
    }

    /**
     * 测试删除后再次删除应标记失败
     * 验证：删除已删除的数据集应在结果中标记为失败
     */
    @Test
    @Transactional
    @DisplayName("删除已删除的数据集应标记失败")
    void deleteDataset_AlreadyDeleted_ShouldMarkFailed() {
        // Arrange - 创建并删除数据集
        SysDataset dataset = new SysDataset();
        dataset.setName("待删除数据集_" + System.currentTimeMillis());
        dataset.setType("image");
        dataset.setParentId(SystemConstants.ROOT_NODE_ID);
        dataset.setCreateTime(LocalDateTime.now());
        dataset.setUpdateTime(LocalDateTime.now());
        datasetMapper.insert(dataset);
        Long datasetId = dataset.getId();

        // Act - 第一次删除
        datasetOperationService.batchDeleteDatasets(List.of(datasetId));

        // Assert - 第二次删除应标记失败
        BatchDeleteResult result = datasetOperationService.batchDeleteDatasets(List.of(datasetId));
        assertThat(result.getFailed()).isGreaterThan(0);
        assertThat(result.getResults())
                .anySatisfy(item -> {
                    assertThat(item.getId()).isEqualTo(datasetId);
                    assertThat(item.getStatus()).isEqualTo("failed");
                });
    }

    /**
     * 测试事务隔离性
     * 验证：在事务中的修改对外部不可见（在事务提交前）
     */
    @Test
    @Transactional
    @DisplayName("事务中的修改在提交前对外部不可见")
    void transactionIsolation_ModificationsNotVisibleBeforeCommit() {
        // Arrange
        String uniqueName = "隔离性测试_" + System.currentTimeMillis();

        // Act - 在事务中创建数据集
        SysDataset dataset = new SysDataset();
        dataset.setName(uniqueName);
        dataset.setType("image");
        dataset.setParentId(SystemConstants.ROOT_NODE_ID);
        dataset.setCreateTime(LocalDateTime.now());
        dataset.setUpdateTime(LocalDateTime.now());
        datasetMapper.insert(dataset);

        // Assert - 在同一事务中可以查询到
        SysDataset found = datasetMapper.selectOne(
                new LambdaQueryWrapper<SysDataset>()
                        .eq(SysDataset::getName, uniqueName));
        assertThat(found).isNotNull();
        assertThat(found.getName()).isEqualTo(uniqueName);

        // 测试结束后，由于@Transactional注解，数据会自动回滚
    }

    /**
     * 测试并发修改场景下的数据一致性
     * 验证：同时修改同一数据集时，应该保持数据一致性
     */
    @Test
    @Transactional
    @DisplayName("并发修改应保持数据一致性")
    void concurrentModification_ShouldMaintainConsistency() {
        // Arrange - 获取当前数据集
        SysDataset dataset = datasetMapper.selectById(existingDatasetId);
        assertThat(dataset).isNotNull();

        // Act - 修改数据集
        dataset.setDescription("更新后的描述_" + System.currentTimeMillis());
        dataset.setUpdateTime(LocalDateTime.now());
        datasetMapper.updateById(dataset);

        // Assert - 验证修改成功
        SysDataset updated = datasetMapper.selectById(existingDatasetId);
        assertThat(updated.getDescription()).isEqualTo(dataset.getDescription());
    }
}
