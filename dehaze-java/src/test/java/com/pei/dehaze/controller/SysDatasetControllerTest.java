package com.pei.dehaze.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.form.BatchDeleteRequest;
import com.pei.dehaze.model.form.DatasetAddForm;
import com.pei.dehaze.model.form.DatasetUpdateForm;
import com.pei.dehaze.model.vo.BatchDeleteResult;
import com.pei.dehaze.model.vo.DatasetVO;
import com.pei.dehaze.service.DatasetOperationService;
import com.pei.dehaze.service.SysDatasetService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

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
 * 数据集控制器单元测试
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@WebMvcTest(SysDatasetController.class)
@AutoConfigureMockMvc(addFilters = false)
@DisplayName("数据集接口测试")
class SysDatasetControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private SysDatasetService datasetService;

    @MockBean
    private DatasetOperationService datasetOperationService;

    @Test
    @DisplayName("GET /api/v1/datasets - 获取数据集列表")
    void testListDatasets() throws Exception {
        // Given
        DatasetVO dataset = new DatasetVO();
        dataset.setId(1L);
        dataset.setName("测试数据集");
        List<DatasetVO> datasets = Collections.singletonList(dataset);

        when(datasetService.getList(any())).thenReturn(datasets);

        // When & Then
        mockMvc.perform(get("/api/v1/datasets")
                        .param("keyword", "测试")
                        .param("pageNum", "1")
                        .param("pageSize", "20"))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data").isArray())
                .andExpect(jsonPath("$.data[0].id").value(1))
                .andExpect(jsonPath("$.data[0].name").value("测试数据集"));
    }

    @Test
    @DisplayName("GET /api/v1/datasets/{id} - 获取数据集详情")
    void testGetDatasetById() throws Exception {
        // Given
        Long datasetId = 1L;
        DatasetVO dataset = new DatasetVO();
        dataset.setId(datasetId);
        dataset.setName("户外场景数据集");

        when(datasetService.getDatasetById(datasetId)).thenReturn(dataset);

        // When & Then
        mockMvc.perform(get("/api/v1/datasets/{id}", datasetId))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.id").value(1))
                .andExpect(jsonPath("$.data.name").value("户外场景数据集"));
    }

    @Test
    @DisplayName("POST /api/v1/datasets - 创建数据集")
    void testCreateDataset() throws Exception {
        // Given
        DatasetAddForm form = new DatasetAddForm();
        form.setName("新建数据集");
        form.setType("USER");
        form.setDescription("测试描述");
        form.setParentId(0L);

        DatasetVO createdDataset = new DatasetVO();
        createdDataset.setId(1L);
        createdDataset.setName("新建数据集");

        when(datasetService.addDataset(any())).thenReturn(createdDataset);

        // When & Then
        mockMvc.perform(post("/api/v1/datasets")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form)))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.id").value(1))
                .andExpect(jsonPath("$.data.name").value("新建数据集"));
    }

    @Test
    @DisplayName("PUT /api/v1/datasets/{id} - 更新数据集")
    void testUpdateDataset() throws Exception {
        // Given
        Long datasetId = 1L;
        DatasetUpdateForm form = new DatasetUpdateForm();
        form.setName("更新后的数据集");
        form.setDescription("更新后的描述");

        DatasetVO updatedDataset = new DatasetVO();
        updatedDataset.setId(datasetId);
        updatedDataset.setName("更新后的数据集");

        when(datasetService.updateDataset(eq(datasetId), any())).thenReturn(updatedDataset);

        // When & Then
        mockMvc.perform(put("/api/v1/datasets/{id}", datasetId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form)))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.id").value(1))
                .andExpect(jsonPath("$.data.name").value("更新后的数据集"));
    }

    @Test
    @DisplayName("DELETE /api/v1/datasets/{id} - 删除单个数据集")
    void testDeleteDataset() throws Exception {
        // Given
        Long datasetId = 1L;

        // When & Then
        mockMvc.perform(delete("/api/v1/datasets/{id}", datasetId))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @Test
    @DisplayName("DELETE /api/v1/datasets/batch - 批量删除数据集")
    void testBatchDeleteDatasets() throws Exception {
        // Given
        BatchDeleteRequest request = new BatchDeleteRequest();
        request.setIds(Arrays.asList(1L, 2L, 3L));

        BatchDeleteResult result = BatchDeleteResult.builder()
                .total(3)
                .succeeded(2)
                .failed(1)
                .results(Arrays.asList(
                        BatchDeleteResult.DeleteResultItem.builder()
                                .id(1L)
                                .status("success")
                                .build(),
                        BatchDeleteResult.DeleteResultItem.builder()
                                .id(2L)
                                .status("success")
                                .build(),
                        BatchDeleteResult.DeleteResultItem.builder()
                                .id(3L)
                                .status("failed")
                                .message("数据集不存在")
                                .errorCode("RESOURCE_NOT_FOUND")
                                .build()
                ))
                .build();

        when(datasetOperationService.batchDeleteDatasets(anyList())).thenReturn(result);

        // When & Then
        mockMvc.perform(delete("/api/v1/datasets/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.total").value(3))
                .andExpect(jsonPath("$.data.succeeded").value(2))
                .andExpect(jsonPath("$.data.failed").value(1))
                .andExpect(jsonPath("$.data.results").isArray())
                .andExpect(jsonPath("$.data.results[2].status").value("failed"))
                .andExpect(jsonPath("$.data.results[2].errorCode").value("RESOURCE_NOT_FOUND"));
    }

    @Test
    @DisplayName("GET /api/v1/datasets - 带筛选条件查询")
    void testListDatasetsWithFilters() throws Exception {
        // Given
        when(datasetService.getList(any())).thenReturn(Collections.emptyList());

        // When & Then
        mockMvc.perform(get("/api/v1/datasets")
                        .param("keyword", "测试")
                        .param("type", "training")
                        .param("status", "1")
                        .param("pageNum", "1")
                        .param("pageSize", "20"))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data").isArray());
    }

    // ==================== 异常场景测试 ====================

    @Test
    @DisplayName("GET /api/v1/datasets/{id} - 数据集不存在返回400")
    void testGetDatasetById_NotFound() throws Exception {
        // Given
        Long datasetId = 999L;
        when(datasetService.getDatasetById(datasetId))
                .thenThrow(new BusinessException("数据集不存在"));

        // When & Then - BusinessException 默认返回 400
        mockMvc.perform(get("/api/v1/datasets/{id}", datasetId))
                .andDo(print())
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("B0001"));
    }

    @Test
    @DisplayName("POST /api/v1/datasets - name为空返回400")
    void testCreateDataset_NameEmpty() throws Exception {
        // Given
        DatasetAddForm form = new DatasetAddForm();
        form.setName("");
        form.setType("USER");
        form.setDescription("测试描述");
        form.setParentId(0L);

        // When & Then - @NotBlank校验会拒绝空字符串
        mockMvc.perform(post("/api/v1/datasets")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form)))
                .andDo(print())
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("POST /api/v1/datasets - name为null返回400")
    void testCreateDataset_NameNull() throws Exception {
        // Given
        DatasetAddForm form = new DatasetAddForm();
        form.setName(null);
        form.setType("USER");
        form.setDescription("测试描述");
        form.setParentId(0L);

        // When & Then - @NotBlank校验会拒绝null值
        mockMvc.perform(post("/api/v1/datasets")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form)))
                .andDo(print())
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("POST /api/v1/datasets - 名称重复返回异常")
    void testCreateDataset_DuplicateName() throws Exception {
        // Given
        DatasetAddForm form = new DatasetAddForm();
        form.setName("已存在的名称");
        form.setType("USER");
        form.setDescription("测试描述");
        form.setParentId(0L);

        when(datasetService.addDataset(any()))
                .thenThrow(new BusinessException("数据集名称已存在"));

        // When & Then - BusinessException 默认返回 400
        mockMvc.perform(post("/api/v1/datasets")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form)))
                .andDo(print())
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("B0001"));
    }

    @Test
    @DisplayName("PUT /api/v1/datasets/{id} - 数据集不存在返回400")
    void testUpdateDataset_NotFound() throws Exception {
        // Given
        Long datasetId = 999L;
        DatasetUpdateForm form = new DatasetUpdateForm();
        form.setName("更新后的数据集");

        when(datasetService.updateDataset(eq(datasetId), any()))
                .thenThrow(new BusinessException("数据集不存在"));

        // When & Then
        mockMvc.perform(put("/api/v1/datasets/{id}", datasetId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form)))
                .andDo(print())
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("B0001"));
    }

    @Test
    @DisplayName("DELETE /api/v1/datasets/{id} - 数据集不存在返回400")
    void testDeleteDataset_NotFound() throws Exception {
        // Given
        Long datasetId = 999L;
        doThrow(new BusinessException("数据集不存在"))
                .when(datasetService).deleteDataset(datasetId);

        // When & Then
        mockMvc.perform(delete("/api/v1/datasets/{id}", datasetId))
                .andDo(print())
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("B0001"));
    }
}
