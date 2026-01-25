package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.form.*;
import com.pei.dehaze.model.query.DatasetItemQuery;
import com.pei.dehaze.model.vo.*;
import com.pei.dehaze.service.DatasetOperationService;
import com.pei.dehaze.service.SysDatasetItemService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.doThrow;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultHandlers.print;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * 数据项控制器单元测试
 * 测试目的：验证数据项管理接口的正确性
 * 测试范围：
 * 1. 基础CRUD操作（查询、创建、更新、删除）
 * 2. 上传相关接口（单个上传、批量上传）
 * 3. 下载相关接口（单个下载、批量下载）
 * 4. 异常场景处理
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@WebMvcTest(SysDatasetItemController.class)
@AutoConfigureMockMvc(addFilters = false)
@DisplayName("数据项接口测试")
class SysDatasetItemControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private SysDatasetItemService sysDatasetItemService;

    @MockBean
    private DatasetOperationService datasetOperationService;

    // ==================== 基础CRUD测试 ====================

    /**
     * 测试获取数据项详情
     * 测试场景：根据ID获取数据项完整信息
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回数据包含基本信息、清晰图、有雾图列表
     */
    @Test
    @DisplayName("GET /api/v1/dataset-items/{id} - 获取数据项详情")
    void testGetDatasetItemById() throws Exception {
        // Arrange
        Long itemId = 1L;
        DatasetItemVO itemVO = createMockDatasetItemVO(itemId);

        when(sysDatasetItemService.getDatasetItem(itemId)).thenReturn(itemVO);

        // Act & Assert
        mockMvc.perform(get("/api/v1/dataset-items/{id}", itemId))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.id").value(1))
                .andExpect(jsonPath("$.data.name").value("城市街道_001"))
                .andExpect(jsonPath("$.data.sceneType").value("outdoor"))
                .andExpect(jsonPath("$.data.description").value("城市主干道雾霾场景"))
                .andExpect(jsonPath("$.data.usageCount").value(15))
                .andExpect(jsonPath("$.data.clearImage").exists())
                .andExpect(jsonPath("$.data.clearImage.id").value(101))
                .andExpect(jsonPath("$.data.hazyImages").isArray())
                .andExpect(jsonPath("$.data.hazyImages[0].id").value(102))
                .andExpect(jsonPath("$.data.hazyImages[0].hazeLevel").value("light"));
    }

    /**
     * 测试分页查询数据项列表
     * 测试场景：根据查询条件分页获取数据项
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回分页数据结构正确
     * 3. 查询参数正确传递
     */
    @Test
    @DisplayName("GET /api/v1/dataset-items - 分页查询数据项列表")
    void testListDatasetItems() throws Exception {
        // Arrange
        Page<DatasetItemVO> page = new Page<>(1, 20);
        List<DatasetItemVO> records = new ArrayList<>();
        records.add(createMockDatasetItemVO(1L));
        records.add(createMockDatasetItemVO(2L));
        page.setRecords(records);
        page.setTotal(2);

        when(sysDatasetItemService.pageSearchDatasetItems(any(DatasetItemQuery.class)))
                .thenReturn(page);

        // Act & Assert
        mockMvc.perform(get("/api/v1/dataset-items")
                        .param("datasetId", "10")
                        .param("keyword", "城市")
                        .param("sceneType", "outdoor")
                        .param("pageNum", "1")
                        .param("pageSize", "20"))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.list").isArray())
                .andExpect(jsonPath("$.data.list[0].id").value(1))
                .andExpect(jsonPath("$.data.total").value(2));
    }

    /**
     * 测试创建空数据项
     * 测试场景：创建一个不包含图片的数据项
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回创建的数据项信息
     */
    @Test
    @DisplayName("POST /api/v1/dataset-items - 创建空数据项")
    void testAddItem() throws Exception {
        // Arrange
        DatasetItemCreateForm form = new DatasetItemCreateForm();
        form.setDatasetId(10L);
        form.setName("城市街道_001");
        form.setSceneType("outdoor");
        form.setDescription("城市主干道雾霾场景");

        DatasetItemVO createdItem = new DatasetItemVO();
        createdItem.setId(1L);
        createdItem.setDatasetId(10L);
        createdItem.setName("城市街道_001");
        createdItem.setCreateTime(LocalDateTime.now());

        when(sysDatasetItemService.createAndReturnDatasetItem(anyLong(), anyString()))
                .thenReturn(createdItem);

        // Act & Assert
        mockMvc.perform(post("/api/v1/dataset-items")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form)))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.id").value(1))
                .andExpect(jsonPath("$.data.name").value("城市街道_001"));
    }

    /**
     * 测试更新数据项信息
     * 测试场景：更新数据项的基本信息
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回更新后的数据项信息
     */
    @Test
    @DisplayName("PUT /api/v1/dataset-items/{id} - 更新数据项信息")
    void testUpdateItem() throws Exception {
        // Arrange
        Long itemId = 1L;
        DatasetItemUpdateForm form = new DatasetItemUpdateForm();
        form.setName("城市街道_001_v2");
        form.setSceneType("indoor");

        DatasetItemVO updatedItem = new DatasetItemVO();
        updatedItem.setId(itemId);
        updatedItem.setName("城市街道_001_v2");
        updatedItem.setSceneType("indoor");
        updatedItem.setUpdateTime(LocalDateTime.now());

        when(sysDatasetItemService.updateAndReturnDatasetItem(eq(itemId), anyString(), anyString()))
                .thenReturn(updatedItem);

        // Act & Assert
        mockMvc.perform(put("/api/v1/dataset-items/{id}", itemId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form)))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.id").value(1))
                .andExpect(jsonPath("$.data.name").value("城市街道_001_v2"))
                .andExpect(jsonPath("$.data.sceneType").value("indoor"));
    }

    /**
     * 测试删除单个数据项
     * 测试场景：删除指定ID的数据项
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回成功标识
     */
    @Test
    @DisplayName("DELETE /api/v1/dataset-items/{id} - 删除数据项")
    void testRemoveItem() throws Exception {
        // Arrange
        Long itemId = 1L;

        // Act & Assert
        mockMvc.perform(delete("/api/v1/dataset-items/{id}", itemId))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    /**
     * 测试批量删除数据项
     * 测试场景：批量删除多个数据项
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回批量操作结果
     */
    @Test
    @DisplayName("DELETE /api/v1/dataset-items/batch - 批量删除数据项")
    void testBatchDeleteDatasetItems() throws Exception {
        // Arrange
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(Arrays.asList(1L, 2L, 3L));

        BatchOperationResultVO result = new BatchOperationResultVO();
        result.setSuccessCount(2);
        result.setFailedCount(1);
        result.setMessage("批量删除完成：成功2个，失败1个");

        when(datasetOperationService.batchDeleteDatasetItemsCascadeWithResult(anyList()))
                .thenReturn(result);

        // Act & Assert
        mockMvc.perform(delete("/api/v1/dataset-items/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form)))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.successCount").value(2))
                .andExpect(jsonPath("$.data.failedCount").value(1));
    }

    // ==================== 上传相关测试 ====================

    /**
     * 测试创建数据项并上传配对图片
     * 测试场景：一步完成数据项创建和图片上传
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回包含图片信息的数据项
     */
    @Test
    @DisplayName("POST /api/v1/dataset-items/upload - 创建数据项并上传配对图片")
    void testUploadImagePair() throws Exception {
        // Arrange
        DatasetItemVO result = createMockDatasetItemVO(1L);

        when(datasetOperationService.createDatasetItemWithImages(any(DatasetItemUploadForm.class)))
                .thenReturn(result);

        MockMultipartFile clearImage = new MockMultipartFile(
                "clearImage", "clear.jpg", "image/jpeg", "clear image content".getBytes());
        MockMultipartFile hazyImage = new MockMultipartFile(
                "hazyImages", "hazy_light.jpg", "image/jpeg", "hazy image content".getBytes());

        // Act & Assert
        mockMvc.perform(multipart("/api/v1/dataset-items/upload")
                        .file(clearImage)
                        .file(hazyImage)
                        .param("datasetId", "10")
                        .param("name", "城市街道_001")
                        .param("sceneType", "outdoor")
                        .param("hazeLevels", "light"))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.id").value(1))
                .andExpect(jsonPath("$.data.clearImage").exists())
                .andExpect(jsonPath("$.data.hazyImages").isArray());
    }

    /**
     * 测试批量创建数据项并上传图片
     * 测试场景：批量上传多个数据项的配对图片
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回批量上传结果
     */
    @Test
    @DisplayName("POST /api/v1/dataset-items/batch - 批量创建数据项并上传图片")
    void testBatchUploadImagePairs() throws Exception {
        // Arrange
        BatchUploadResultVO result = new BatchUploadResultVO();
        result.setTotal(10);
        result.setSucceeded(8);
        result.setFailed(2);

        BatchUploadSuccessItemVO successItem = new BatchUploadSuccessItemVO();
        successItem.setId(1L);
        successItem.setName("street_001");
        successItem.setFileCount(3);
        result.setSuccessItems(Collections.singletonList(successItem));

        BatchUploadFailedItemVO failedItem = new BatchUploadFailedItemVO();
        failedItem.setFileName("invalid.jpg");
        failedItem.setReason("未找到配对的清晰图");
        result.setFailedItems(Collections.singletonList(failedItem));

        when(datasetOperationService.batchCreateDatasetItemsWithImages(any(BatchDatasetItemUploadForm.class)))
                .thenReturn(result);

        MockMultipartFile file1 = new MockMultipartFile(
                "files", "street_001_clear.jpg", "image/jpeg", "content1".getBytes());
        MockMultipartFile file2 = new MockMultipartFile(
                "files", "street_001_hazy_light.jpg", "image/jpeg", "content2".getBytes());

        // Act & Assert
        mockMvc.perform(multipart("/api/v1/dataset-items/batch")
                        .file(file1)
                        .file(file2)
                        .param("datasetId", "10"))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.total").value(10))
                .andExpect(jsonPath("$.data.succeeded").value(8))
                .andExpect(jsonPath("$.data.failed").value(2))
                .andExpect(jsonPath("$.data.successItems").isArray())
                .andExpect(jsonPath("$.data.failedItems").isArray());
    }

    // ==================== 异常场景测试 ====================

    /**
     * 测试获取不存在的数据项
     * 测试场景：查询不存在的数据项ID
     * 验证内容：
     * 1. 返回状态码400
     * 2. 返回错误信息
     */
    @Test
    @DisplayName("GET /api/v1/dataset-items/{id} - 数据项不存在返回400")
    void testGetDatasetItemById_NotFound() throws Exception {
        // Arrange
        Long itemId = 999L;
        when(sysDatasetItemService.getDatasetItem(itemId))
                .thenThrow(new BusinessException("数据项不存在"));

        // Act & Assert
        mockMvc.perform(get("/api/v1/dataset-items/{id}", itemId))
                .andDo(print())
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("B0001"));
    }

    /**
     * 测试更新不存在的数据项
     * 测试场景：更新不存在的数据项
     * 验证内容：
     * 1. 返回状态码2 00
     * 2. Service层正常返回null，不抛出异常
     */
    @Test
    @DisplayName("PUT /api/v1/dataset-items/{id} - 数据项不存在返回200")
    void testUpdateItem_NotFound() throws Exception {
        // Arrange
        Long itemId = 999L;
        DatasetItemUpdateForm form = new DatasetItemUpdateForm();
        form.setName("更新后的名称");

        // Service层返回null而不是抛出异常
        when(sysDatasetItemService.updateAndReturnDatasetItem(eq(itemId), anyString(), anyString()))
                .thenReturn(null);

        // Act & Assert
        mockMvc.perform(put("/api/v1/dataset-items/{id}", itemId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form)))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data").isEmpty());
    }

    /**
     * 测试删除不存在的数据项
     * 测试场景：删除不存在的数据项
     * 验证内容：
     * 1. 返回状态码400
     * 2. 返回错误信息
     */
    @Test
    @DisplayName("DELETE /api/v1/dataset-items/{id} - 数据项不存在返回400")
    void testRemoveItem_NotFound() throws Exception {
        // Arrange
        Long itemId = 999L;
        doThrow(new BusinessException("数据项不存在"))
                .when(datasetOperationService).deleteDatasetItemCascade(itemId);

        // Act & Assert
        mockMvc.perform(delete("/api/v1/dataset-items/{id}", itemId))
                .andDo(print())
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("B0001"));
    }

    /**
     * 测试分页查询（带多个筛选条件）
     * 测试场景：使用多个筛选条件查询
     * 验证内容：
     * 1. 返回状态码200
     * 2. 查询参数正确传递
     */
    @Test
    @DisplayName("GET /api/v1/dataset-items - 带多个筛选条件查询")
    void testListDatasetItems_WithMultipleFilters() throws Exception {
        // Arrange
        Page<DatasetItemVO> page = new Page<>(1, 20);
        page.setRecords(Collections.emptyList());
        page.setTotal(0);

        when(sysDatasetItemService.pageSearchDatasetItems(any(DatasetItemQuery.class)))
                .thenReturn(page);

        // Act & Assert
        mockMvc.perform(get("/api/v1/dataset-items")
                        .param("datasetId", "10")
                        .param("keyword", "城市")
                        .param("sceneType", "outdoor")
                        .param("hazeLevel", "light")
                        .param("minWidth", "1920")
                        .param("maxWidth", "3840")
                        .param("minHeight", "1080")
                        .param("maxHeight", "2160")
                        .param("sortBy", "createTime")
                        .param("sortOrder", "desc")
                        .param("pageNum", "1")
                        .param("pageSize", "20"))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.list").isArray());
    }

    // ==================== 辅助方法 ====================

    /**
     * 创建模拟的DatasetItemVO对象
     */
    private DatasetItemVO createMockDatasetItemVO(Long id) {
        DatasetItemVO itemVO = new DatasetItemVO();
        itemVO.setId(id);
        itemVO.setDatasetId(10L);
        itemVO.setName("城市街道_001");
        itemVO.setSceneType("outdoor");
        itemVO.setDescription("城市主干道雾霾场景");
        itemVO.setUsageCount(15);
        itemVO.setImageCount(3);
        itemVO.setCreateTime(LocalDateTime.now());
        itemVO.setUpdateTime(LocalDateTime.now());

        ImageUrlVO clearImage = new ImageUrlVO();
        clearImage.setId(101L);
        clearImage.setType("clear");
        clearImage.setUrl("https://cdn.example.com/clear/001.jpg");
        clearImage.setThumbnailUrl("https://cdn.example.com/clear/thumb_001.jpg");
        clearImage.setWidth(1920);
        clearImage.setHeight(1080);
        clearImage.setSizeBytes(2560000L);
        clearImage.setFormattedSize("2.44MB");
        clearImage.setFormat("jpg");
        clearImage.setMd5("abc123");
        itemVO.setClearImage(clearImage);

        List<ImageUrlVO> hazyImages = new ArrayList<>();
        ImageUrlVO hazyImage = new ImageUrlVO();
        hazyImage.setId(102L);
        hazyImage.setType("hazy");
        hazyImage.setHazeLevel("light");
        hazyImage.setUrl("https://cdn.example.com/hazy/001_light.jpg");
        hazyImage.setThumbnailUrl("https://cdn.example.com/hazy/thumb_001_light.jpg");
        hazyImage.setWidth(1920);
        hazyImage.setHeight(1080);
        hazyImage.setSizeBytes(2800000L);
        hazyImage.setFormattedSize("2.67MB");
        hazyImages.add(hazyImage);
        itemVO.setHazyImages(hazyImages);

        return itemVO;
    }
}
