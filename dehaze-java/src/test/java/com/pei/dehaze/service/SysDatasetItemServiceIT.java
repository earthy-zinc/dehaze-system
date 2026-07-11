package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.query.DatasetItemQuery;
import com.pei.dehaze.model.vo.BatchOperationResultVO;
import com.pei.dehaze.model.vo.DatasetItemVO;
import com.pei.dehaze.mapper.SysDatasetItemMapper;
import com.pei.dehaze.mapper.SysDatasetMapper;
import com.pei.dehaze.mapper.SysFileMapper;
import com.pei.dehaze.mapper.SysItemFileMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 数据项服务集成测试
 * 测试目的：验证数据项的完整业务流程和跨模块交互
 * 测试范围：数据项CRUD操作、批量操作、分页查询、与文件模块的关联
 */
@SpringBootTest
@DisplayName("数据项服务集成测试")
class SysDatasetItemServiceIT {

    @Autowired
    private SysDatasetItemService datasetItemService;

    @Autowired
    private DatasetOperationService datasetOperationService;

    @Autowired
    private SysDatasetItemMapper datasetItemMapper;

    @Autowired
    private SysDatasetMapper datasetMapper;

    @Autowired
    private SysFileMapper fileMapper;

    @Autowired
    private SysItemFileMapper itemFileMapper;

    private SysDataset testDataset;

    @BeforeEach
    void setUp() {
        // 创建测试数据集
        testDataset = new SysDataset();
        testDataset.setName("测试数据集");
        testDataset.setParentId(0L);
        testDataset.setType("FOLDER");
        datasetMapper.insert(testDataset);
    }

    /**
     * 测试目的：验证创建数据项的基本功能
     * 测试场景：创建单个数据项并验证数据库存储
     * 验证内容：数据项正确保存，字段值正确设置
     */
    @Test
    @DisplayName("创建数据项-成功")
    @Transactional
    void testCreateDatasetItem_Success() {
        SysDatasetItem result = datasetItemService.createDatasetItem(testDataset.getId());

        assertNotNull(result);
        assertNotNull(result.getId());
        assertEquals(testDataset.getId(), result.getDatasetId());

        SysDatasetItem fromDb = datasetItemMapper.selectById(result.getId());
        assertNotNull(fromDb);
        assertEquals(testDataset.getId(), fromDb.getDatasetId());
    }

    /**
     * 测试目的：验证创建带名称的数据项功能
     * 测试场景：创建带名称的数据项
     * 验证内容：数据项名称正确保存
     */
    @Test
    @DisplayName("创建带名称的数据项-成功")
    @Transactional
    void testCreateDatasetItemWithName_Success() {
        String itemName = "测试数据项";
        SysDatasetItem result = datasetItemService.createDatasetItem(testDataset.getId(), itemName);

        assertNotNull(result);
        assertEquals(itemName, result.getName());

        SysDatasetItem fromDb = datasetItemMapper.selectById(result.getId());
        assertNotNull(fromDb);
        assertEquals(itemName, fromDb.getName());
    }

    /**
     * 测试目的：验证创建并返回数据项VO功能
     * 测试场景：创建数据项并返回完整VO对象
     * 验证内容：返回的VO包含正确的数据项信息
     */
    @Test
    @DisplayName("创建并返回数据项VO-成功")
    @Transactional
    void testCreateAndReturnDatasetItem_Success() {
        String itemName = "测试数据项";
        DatasetItemVO result = datasetItemService.createAndReturnDatasetItem(testDataset.getId(), itemName);

        assertNotNull(result);
        assertNotNull(result.getId());
        assertEquals(testDataset.getId(), result.getDatasetId());
        assertEquals(itemName, result.getName());
    }

    /**
     * 测试目的：验证删除数据项功能
     * 测试场景：创建数据项后删除
     * 验证内容：数据项正确删除，不再存在于数据库
     */
    @Test
    @DisplayName("删除数据项-成功")
    @Transactional
    void testDeleteDatasetItem_Success() {
        SysDatasetItem item = datasetItemService.createDatasetItem(testDataset.getId());

        datasetItemService.deleteDatasetItem(item.getId());

        SysDatasetItem deleted = datasetItemMapper.selectById(item.getId());
        assertNull(deleted);
    }

    /**
     * 测试目的：验证更新数据项功能
     * 测试场景：创建数据项后修改其名称
     * 验证内容：数据项名称正确更新
     */
    @Test
    @DisplayName("更新数据项-成功")
    @Transactional
    void testUpdateDatasetItem_Success() {
        SysDatasetItem item = datasetItemService.createDatasetItem(testDataset.getId(), "原始名称");

        String newName = "更新后名称";
        datasetItemService.updateDatasetItem(item.getId(), newName);

        SysDatasetItem updated = datasetItemMapper.selectById(item.getId());
        assertNotNull(updated);
        assertEquals(newName, updated.getName());
    }

    /**
     * 测试目的：验证更新并返回数据项VO功能
     * 测试场景：更新数据项名称并返回完整VO对象
     * 验证内容：返回的VO包含更新后的信息
     */
    @Test
    @DisplayName("更新并返回数据项VO-成功")
    @Transactional
    void testUpdateAndReturnDatasetItem_Success() {
        SysDatasetItem item = datasetItemService.createDatasetItem(testDataset.getId(), "原始名称");

        String newName = "更新后名称";
        DatasetItemVO result = datasetItemService.updateAndReturnDatasetItem(item.getId(), newName, null);

        assertNotNull(result);
        assertEquals(newName, result.getName());
    }

    /**
     * 测试目的：验证批量删除数据项功能
     * 测试场景：批量创建后批量删除
     * 验证内容：批量删除结果正确，删除成功数量匹配
     */
    @Test
    @DisplayName("批量删除数据项-成功")
    @Transactional
    void testBatchDeleteDatasetItems_Success() {
        SysDatasetItem item1 = datasetItemService.createDatasetItem(testDataset.getId());
        SysDatasetItem item2 = datasetItemService.createDatasetItem(testDataset.getId());

        BatchOperationResultVO result = datasetOperationService.batchDeleteDatasetItemsCascadeWithResult(
                List.of(item1.getId(), item2.getId()));

        assertNotNull(result);
        assertEquals(2, result.getSuccessCount());
        assertEquals(0, result.getFailedCount());

        SysDatasetItem deleted1 = datasetItemMapper.selectById(item1.getId());
        SysDatasetItem deleted2 = datasetItemMapper.selectById(item2.getId());
        assertNull(deleted1);
        assertNull(deleted2);
    }

    /**
     * 测试目的：验证分页搜索数据项功能
     * 测试场景：创建多个数据项后分页查询
     * 验证内容：分页结果正确，总数和记录数匹配
     */
    @Test
    @DisplayName("分页搜索数据项-成功")
    @Transactional
    void testPageSearchDatasetItems_Success() {
        datasetItemService.createDatasetItem(testDataset.getId(), "数据项1");
        datasetItemService.createDatasetItem(testDataset.getId(), "数据项2");

        DatasetItemQuery query = new DatasetItemQuery();
        query.setDatasetId(testDataset.getId());
        query.setPageNum(1);
        query.setPageSize(10);

        Page<DatasetItemVO> page = datasetItemService.pageSearchDatasetItems(query);

        assertNotNull(page);
        assertTrue(page.getRecords().size() >= 2);
    }

    /**
     * 测试目的：验证获取数据项详情功能
     * 测试场景：创建数据项后通过ID查询详情
     * 验证内容：返回的VO包含完整的 数据项信息
     */
    @Test
    @DisplayName("获取数据项详情-成功")
    @Transactional
    void testGetDatasetItem_Success() {
        String itemName = "测试数据项";
        SysDatasetItem item = datasetItemService.createDatasetItem(testDataset.getId(), itemName);

        DatasetItemVO result = datasetItemService.getDatasetItem(item.getId());

        assertNotNull(result);
        assertEquals(item.getId(), result.getId());
        assertEquals(itemName, result.getName());
        assertEquals(testDataset.getId(), result.getDatasetId());
    }

    /**
     * 测试目的：验证获取不存在数据项的异常处理
     * 测试场景：查询不存在的数据项ID
     * 验证内容：抛出BusinessException异常
     */
    @Test
    @DisplayName("获取不存在数据项-抛出异常")
    void testGetDatasetItem_NotFound() {
        Long nonExistentId = 999999L;

        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> datasetItemService.getDatasetItem(nonExistentId));

        assertTrue(exception.getMessage().contains("数据项不存在"));
    }

    /**
     * 测试目的：验证数据项与数据集的关联关系
     * 测试场景：创建数据项后验证数据集关联正确
     * 验证内容：数据项的datasetId字段正确指向对应的数据集
     */
    @Test
    @DisplayName("数据项与数据集关联-成功")
    @Transactional
    void testItemDatasetAssociation_Success() {
        SysDatasetItem item = datasetItemService.createDatasetItem(testDataset.getId());

        assertNotNull(item);
        assertEquals(testDataset.getId(), item.getDatasetId());

        SysDataset dataset = datasetMapper.selectById(item.getDatasetId());
        assertNotNull(dataset);
        assertEquals("测试数据集", dataset.getName());
    }

    /**
     * 测试目的：验证批量删除空列表的处理
     * 测试场景：批量删除空的数据项ID列表
     * 验证内容：返回空操作结果
     */
    @Test
    @DisplayName("批量删除空列表-成功")
    void testBatchDeleteDatasetItems_EmptyList() {
        BatchOperationResultVO result = datasetOperationService.batchDeleteDatasetItemsCascadeWithResult(List.of());

        assertNotNull(result);
        assertEquals(0, result.getSuccessCount());
        assertEquals(0, result.getFailedCount());
    }
}
