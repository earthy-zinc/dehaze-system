package com.pei.dehaze.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.form.BatchDeleteForm;
import com.pei.dehaze.model.form.ItemFileUpdateForm;
import com.pei.dehaze.model.vo.BatchDeleteResultVO;
import com.pei.dehaze.model.vo.ImageUrlVO;
import com.pei.dehaze.service.SysDatasetService;
import com.pei.dehaze.service.SysItemFileService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verify;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.csrf;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultHandlers.print;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * 图片文件控制器单元测试
 * 测试目的：验证图片文件管理接口的正确性
 * 测试范围：
 * 1. 图片信息查询（单个查询）
 * 2. 图片上传操作（单个上传）
 * 3. 图片信息修改
 * 4. 图片删除操作（单个删除、批量删除）
 * 5. 异常场景处理
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@WebMvcTest(SysItemFileController.class)
@DisplayName("图片文件接口测试")
class SysItemFileControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private SysItemFileService sysItemFileService;

    @MockBean
    private SysDatasetService sysDatasetService;

    private ImageUrlVO mockImageUrlVO;
    private BatchDeleteResultVO mockBatchDeleteResult;

    @BeforeEach
    void setUp() {
        mockImageUrlVO = createMockImageUrlVO(1L);
        mockBatchDeleteResult = createMockBatchDeleteResult();
    }

    // ==================== 查询相关测试 ====================

    /**
     * 测试获取图片详细信息
     * 测试场景：根据ID获取图片的完整信息，包含配对图片和数据项信息
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回数据包含图片基本信息（URL、缩略图、分辨率、文件大小、格式等）
     * 3. 返回数据包含标注信息（场景类型、雾霾程度、描述等）
     * 4. 返回数据包含配对图片列表
     * 5. 返回数据包含所属数据项简要信息
     */
    @Test
    @WithMockUser
    @DisplayName("GET /api/v1/item-files/{id} - 获取图片详细信息")
    void testGetImageById() throws Exception {
        // Arrange
        Long imageId = 1L;
        when(sysItemFileService.getImageById(imageId)).thenReturn(mockImageUrlVO);

        // Act & Assert
        mockMvc.perform(get("/api/v1/item-files/{id}", imageId))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.id").value(1))
                .andExpect(jsonPath("$.data.itemId").value(10))
                .andExpect(jsonPath("$.data.type").value("clear"))
                .andExpect(jsonPath("$.data.description").value("清晰的城市街道图片"))
                .andExpect(jsonPath("$.data.sceneType").value("outdoor"))
                .andExpect(jsonPath("$.data.hazeLevel").value("none"))
                .andExpect(jsonPath("$.data.width").value(1920))
                .andExpect(jsonPath("$.data.height").value(1080))
                .andExpect(jsonPath("$.data.usageCount").value(5))
                .andExpect(jsonPath("$.data.url").value("https://cdn.example.com/images/clear_001.jpg"))
                .andExpect(jsonPath("$.data.thumbnailUrl").value("https://cdn.example.com/thumbs/clear_001_thumb.jpg"))
                .andExpect(jsonPath("$.data.fileName").value("clear_001.jpg"))
                .andExpect(jsonPath("$.data.formattedSize").value("2.44MB"))
                .andExpect(jsonPath("$.data.format").value("jpg"))
                .andExpect(jsonPath("$.data.datasetId").value(5))
                .andExpect(jsonPath("$.data.datasetName").value("城市街道数据集"))
                .andExpect(jsonPath("$.data.hasPairedImages").value(true))
                .andExpect(jsonPath("$.data.pairedCount").value(3))
                .andExpect(jsonPath("$.data.pairedFiles").isArray())
                .andExpect(jsonPath("$.data.pairedFiles[0].id").value(2))
                .andExpect(jsonPath("$.data.datasetItem").exists())
                .andExpect(jsonPath("$.data.datasetItem.id").value(10))
                .andExpect(jsonPath("$.data.datasetItem.name").value("数据项_001"));
    }

    /**
     * 测试获取不存在图片的信息
     * 测试场景：查询不存在的图片ID
     * 验证内容：
     * 1. 返回状态码400
     * 2. 返回错误信息
     */
    @Test
    @WithMockUser
    @DisplayName("GET /api/v1/item-files/{id} - 图片不存在返回400")
    void testGetImageById_NotFound() throws Exception {
        // Arrange
        Long imageId = 999L;
        when(sysItemFileService.getImageById(imageId))
                .thenThrow(new BusinessException("图片不存在"));

        // Act & Assert
        mockMvc.perform(get("/api/v1/item-files/{id}", imageId))
                .andDo(print())
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("B0001"));
    }

    /**
     * 测试未登录访问图片详情
     * 测试场景：未登录用户尝试访问图片详情
     * 验证内容：
     * 1. 返回状态码401（未授权）
     */
    @Test
    @DisplayName("GET /api/v1/item-files/{id} - 未登录返回401")
    void testGetImageById_Unauthorized() throws Exception {
        // Arrange
        Long imageId = 1L;

        // Act & Assert
        mockMvc.perform(get("/api/v1/item-files/{id}", imageId))
                .andDo(print())
                .andExpect(status().isUnauthorized());
    }

    // ==================== 上传相关测试 ====================

    /**
     * 测试上传数据项图片
     * 测试场景：向指定的数据项添加图片文件
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回上传后的图片完整信息
     * 3. Service层正确调用保存方法
     */
    @Test
    @WithMockUser
    @DisplayName("POST /api/v1/item-files - 上传数据项图片")
    void testUpload() throws Exception {
        // Arrange
        Long itemId = 10L;
        String datasetName = "城市街道数据集";

        when(sysDatasetService.getDatasetNameByItemId(itemId)).thenReturn(datasetName);
        when(sysItemFileService.saveItemFile(eq(itemId), any())).thenReturn(mockImageUrlVO);

        MockMultipartFile file = new MockMultipartFile(
                "file", "clear_001.jpg", "image/jpeg", "test image content".getBytes());

        // Act & Assert
        mockMvc.perform(multipart("/api/v1/item-files")
                        .file(file)
                        .param("itemId", "10")
                        .param("type", "clear")
                        .param("description", "清晰的城市街道图片")
                        .param("sceneType", "outdoor")
                        .param("hazeLevel", "none")
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.id").value(1))
                .andExpect(jsonPath("$.data.type").value("clear"))
                .andExpect(jsonPath("$.data.description").value("清晰的城市街道图片"))
                .andExpect(jsonPath("$.data.sceneType").value("outdoor"))
                .andExpect(jsonPath("$.data.hazeLevel").value("none"));

        verify(sysItemFileService).saveItemFile(eq(itemId), any());
    }

    /**
     * 测试上传有雾图片
     * 测试场景：上传有雾图片并标注雾霾程度
     * 验证内容：
     * 1. 返回状态码200
     * 2. 雾霾程度正确记录
     */
    @Test
    @WithMockUser
    @DisplayName("POST /api/v1/item-files - 上传有雾图片")
    void testUpload_HazyImage() throws Exception {
        // Arrange
        Long itemId = 10L;
        String datasetName = "城市街道数据集";

        ImageUrlVO hazyVO = createMockImageUrlVO(2L);
        hazyVO.setType("hazy");
        hazyVO.setHazeLevel("light");

        when(sysDatasetService.getDatasetNameByItemId(itemId)).thenReturn(datasetName);
        when(sysItemFileService.saveItemFile(eq(itemId), any())).thenReturn(hazyVO);

        MockMultipartFile file = new MockMultipartFile(
                "file", "hazy_light.jpg", "image/jpeg", "hazy image content".getBytes());

        // Act & Assert
        mockMvc.perform(multipart("/api/v1/item-files")
                        .file(file)
                        .param("itemId", "10")
                        .param("type", "hazy")
                        .param("hazeLevel", "light")
                        .param("sceneType", "outdoor")
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.type").value("hazy"))
                .andExpect(jsonPath("$.data.hazeLevel").value("light"));
    }

    /**
     * 测试未登录上传图片
     * 测试场景：未登录用户尝试上传图片
     * 验证内容：
     * 1. 返回状态码401（未授权）
     */
    @Test
    @DisplayName("POST /api/v1/item-files - 未登录返回401")
    void testUpload_Unauthorized() throws Exception {
        // Arrange
        MockMultipartFile file = new MockMultipartFile(
                "file", "test.jpg", "image/jpeg", "test content".getBytes());

        // Act & Assert
        mockMvc.perform(multipart("/api/v1/item-files")
                        .file(file)
                        .param("itemId", "10")
                        .param("type", "clear")
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isUnauthorized());
    }

    /**
     * 测试上传图片时缺少必填参数
     * 测试场景：上传图片时未提供itemId
     * 验证内容：
     * 1. 返回状态码400（参数校验失败）
     */
    @Test
    @WithMockUser
    @DisplayName("POST /api/v1/item-files - 缺少itemId返回400")
    void testUpload_MissingItemId() throws Exception {
        // Arrange
        MockMultipartFile file = new MockMultipartFile(
                "file", "test.jpg", "image/jpeg", "test content".getBytes());

        // Act & Assert
        mockMvc.perform(multipart("/api/v1/item-files")
                        .file(file)
                        .param("type", "clear")
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isBadRequest());
    }

    // ==================== 修改相关测试 ====================

    /**
     * 测试修改图片信息
     * 测试场景：更新图片的标注信息（类型、场景类型、雾霾程度、描述）
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回操作结果为成功
     * 3. Service层正确调用更新方法
     */
    @Test
    @WithMockUser
    @DisplayName("PUT /api/v1/item-files/{id} - 修改图片信息")
    void testUpdate() throws Exception {
        // Arrange
        Long imageId = 1L;
        when(sysItemFileService.updateItemFileInfo(eq(imageId), any(ItemFileUpdateForm.class)))
                .thenReturn(true);

        ItemFileUpdateForm form = new ItemFileUpdateForm();
        form.setType("hazy");
        form.setSceneType("indoor");
        form.setHazeLevel("moderate");
        form.setDescription("室内中度雾霾图片");

        // Act & Assert
        mockMvc.perform(put("/api/v1/item-files/{id}", imageId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form))
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));

        verify(sysItemFileService).updateItemFileInfo(eq(imageId), any(ItemFileUpdateForm.class));
    }

    /**
     * 测试修改不存在图片的信息
     * 测试场景：修改不存在的图片
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回操作结果为失败
     */
    @Test
    @WithMockUser
    @DisplayName("PUT /api/v1/item-files/{id} - 图片不存在返回200并标记失败")
    void testUpdate_NotFound() throws Exception {
        // Arrange
        Long imageId = 999L;
        when(sysItemFileService.updateItemFileInfo(eq(imageId), any(ItemFileUpdateForm.class)))
                .thenReturn(false);

        ItemFileUpdateForm form = new ItemFileUpdateForm();
        form.setDescription("修改描述");

        // Act & Assert
        mockMvc.perform(put("/api/v1/item-files/{id}", imageId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form))
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("B0001"));
    }

    /**
     * 测试部分字段修改
     * 测试场景：仅修改部分字段，其他字段保持不变
     * 验证内容：
     * 1. 返回状态码200
     * 2. 仅修改指定字段生效
     */
    @Test
    @WithMockUser
    @DisplayName("PUT /api/v1/item-files/{id} - 部分字段修改")
    void testUpdate_PartialFields() throws Exception {
        // Arrange
        Long imageId = 1L;
        when(sysItemFileService.updateItemFileInfo(eq(imageId), any(ItemFileUpdateForm.class)))
                .thenReturn(true);

        ItemFileUpdateForm form = new ItemFileUpdateForm();
        form.setDescription("更新后的描述");
        // 其他字段为null，表示不修改

        // Act & Assert
        mockMvc.perform(put("/api/v1/item-files/{id}", imageId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form))
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    /**
     * 测试未登录修改图片信息
     * 测试场景：未登录用户尝试修改图片信息
     * 验证内容：
     * 1. 返回状态码401（未授权）
     */
    @Test
    @DisplayName("PUT /api/v1/item-files/{id} - 未登录返回401")
    void testUpdate_Unauthorized() throws Exception {
        // Arrange
        Long imageId = 1L;
        ItemFileUpdateForm form = new ItemFileUpdateForm();
        form.setDescription("修改描述");

        // Act & Assert
        mockMvc.perform(put("/api/v1/item-files/{id}", imageId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form))
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isUnauthorized());
    }

    // ==================== 删除相关测试 ====================

    /**
     * 测试删除单个图片
     * 测试场景：删除指定的图片文件
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回操作结果为成功
     * 3. Service层正确调用删除方法
     */
    @Test
    @WithMockUser
    @DisplayName("DELETE /api/v1/item-files/{id} - 删除单个图片")
    void testDelete() throws Exception {
        // Arrange
        Long imageId = 1L;
        when(sysItemFileService.deleteFile(imageId)).thenReturn(true);

        // Act & Assert
        mockMvc.perform(delete("/api/v1/item-files/{id}", imageId)
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));

        verify(sysItemFileService).deleteFile(imageId);
    }

    /**
     * 测试删除不存在的图片
     * 测试场景：删除不存在的图片
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回操作结果为失败
     */
    @Test
    @WithMockUser
    @DisplayName("DELETE /api/v1/item-files/{id} - 图片不存在返回200并标记失败")
    void testDelete_NotFound() throws Exception {
        // Arrange
        Long imageId = 999L;
        when(sysItemFileService.deleteFile(imageId)).thenReturn(false);

        // Act & Assert
        mockMvc.perform(delete("/api/v1/item-files/{id}", imageId)
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("B0001"));
    }

    /**
     * 测试未登录删除图片
     * 测试场景：未登录用户尝试删除图片
     * 验证内容：
     * 1. 返回状态码401（未授权）
     */
    @Test
    @DisplayName("DELETE /api/v1/item-files/{id} - 未登录返回401")
    void testDelete_Unauthorized() throws Exception {
        // Arrange
        Long imageId = 1L;

        // Act & Assert
        mockMvc.perform(delete("/api/v1/item-files/{id}", imageId)
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isUnauthorized());
    }

    /**
     * 测试批量删除图片（全部成功）
     * 测试场景：批量删除多个图片，所有图片都删除成功
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回批量操作结果
     * 3. 成功数量和失败数量正确
     */
    @Test
    @WithMockUser
    @DisplayName("DELETE /api/v1/item-files/batch - 批量删除图片（全部成功）")
    void testBatchDelete_AllSuccess() throws Exception {
        // Arrange
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(Arrays.asList(1L, 2L, 3L));

        when(sysItemFileService.batchDelete(form.getIds())).thenReturn(mockBatchDeleteResult);

        // Act & Assert
        mockMvc.perform(delete("/api/v1/item-files/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form))
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.successCount").value(3))
                .andExpect(jsonPath("$.data.failedCount").value(0))
                .andExpect(jsonPath("$.data.successIds").isArray())
                .andExpect(jsonPath("$.data.successIds[0]").value(1))
                .andExpect(jsonPath("$.data.successIds[1]").value(2))
                .andExpect(jsonPath("$.data.successIds[2]").value(3))
                .andExpect(jsonPath("$.data.failedItems").isArray());

        verify(sysItemFileService).batchDelete(form.getIds());
    }

    /**
     * 测试批量删除图片（部分失败）
     * 测试场景：批量删除多个图片，部分删除成功，部分删除失败
     * 验证内容：
     * 1. 返回状态码200
     * 2. 返回批量操作结果
     * 3. 成功和失败的ID列表正确
     * 4. 失败原因正确记录
     */
    @Test
    @WithMockUser
    @DisplayName("DELETE /api/v1/item-files/batch - 批量删除图片（部分失败）")
    void testBatchDelete_PartialFailure() throws Exception {
        // Arrange
        BatchDeleteResultVO result = new BatchDeleteResultVO();
        result.setSuccessIds(Arrays.asList(1L, 2L));
        result.setFailedItems(new ArrayList<>());
        result.getFailedItems().add(new BatchDeleteResultVO.FailedItem(3L, "图片不存在"));
        result.getFailedItems().add(new BatchDeleteResultVO.FailedItem(4L, "文件删除失败"));
        result.setSuccessCount(2);
        result.setFailedCount(2);

        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(Arrays.asList(1L, 2L, 3L, 4L));

        when(sysItemFileService.batchDelete(form.getIds())).thenReturn(result);

        // Act & Assert
        mockMvc.perform(delete("/api/v1/item-files/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form))
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.successCount").value(2))
                .andExpect(jsonPath("$.data.failedCount").value(2))
                .andExpect(jsonPath("$.data.successIds").isArray())
                .andExpect(jsonPath("$.data.successIds[0]").value(1))
                .andExpect(jsonPath("$.data.failedItems").isArray())
                .andExpect(jsonPath("$.data.failedItems[0].id").value(3))
                .andExpect(jsonPath("$.data.failedItems[0].reason").value("图片不存在"));
    }

    /**
     * 测试批量删除空列表
     * 测试场景：批量删除时传入空ID列表
     * 验证内容：
     * 1. 返回状态码400（验证失败）
     * 2. 返回验证错误信息
     */
    @Test
    @WithMockUser
    @DisplayName("DELETE /api/v1/item-files/batch - 批量删除空列表返回400")
    void testBatchDelete_EmptyList() throws Exception {
        // Arrange
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(Collections.emptyList());

        // Act & Assert
        mockMvc.perform(delete("/api/v1/item-files/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form))
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isBadRequest());
    }

    /**
     * 测试未登录批量删除图片
     * 测试场景：未登录用户尝试批量删除图片
     * 验证内容：
     * 1. 返回状态码401（未授权）
     */
    @Test
    @DisplayName("DELETE /api/v1/item-files/batch - 未登录返回401")
    void testBatchDelete_Unauthorized() throws Exception {
        // Arrange
        BatchDeleteForm form = new BatchDeleteForm();
        form.setIds(Arrays.asList(1L, 2L, 3L));

        // Act & Assert
        mockMvc.perform(delete("/api/v1/item-files/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(form))
                        .with(csrf()))
                .andDo(print())
                .andExpect(status().isUnauthorized());
    }

    // ==================== 辅助方法 ====================

    /**
     * 创建模拟的ImageUrlVO对象
     */
    private ImageUrlVO createMockImageUrlVO(Long id) {
        ImageUrlVO vo = new ImageUrlVO();
        vo.setId(id);
        vo.setItemId(10L);
        vo.setType("clear");
        vo.setDescription("清晰的城市街道图片");
        vo.setSceneType("outdoor");
        vo.setHazeLevel("none");
        vo.setWidth(1920);
        vo.setHeight(1080);
        vo.setUsageCount(5L);
        vo.setUrl("https://cdn.example.com/images/clear_001.jpg");
        vo.setThumbnailUrl("https://cdn.example.com/thumbs/clear_001_thumb.jpg");
        vo.setFileName("clear_001.jpg");
        vo.setFormattedSize("2.44MB");
        vo.setFormat("jpg");
        vo.setDatasetId(5L);
        vo.setDatasetName("城市街道数据集");
        vo.setHasPairedImages(true);
        vo.setPairedCount(3);

        // 设置配对图片列表
        List<com.pei.dehaze.model.vo.SimpleImageUrlVO> pairedFiles = new ArrayList<>();
        for (int i = 2; i <= 3; i++) {
            com.pei.dehaze.model.vo.SimpleImageUrlVO pairedVO = new com.pei.dehaze.model.vo.SimpleImageUrlVO();
            pairedVO.setId((long) i);
            pairedVO.setItemId(10L);
            pairedVO.setDatasetId(5L);
            pairedVO.setType(i == 2 ? "hazy" : "clear");
            pairedVO.setUrl("https://cdn.example.com/images/paired_00" + i + ".jpg");
            pairedVO.setThumbnailUrl("https://cdn.example.com/thumbs/paired_00" + i + "_thumb.jpg");
            pairedVO.setFileName("paired_00" + i + ".jpg");
            pairedVO.setFormattedSize("2.50MB");
            pairedVO.setFormat("jpg");
            pairedVO.setDescription(i == 2 ? "轻度雾霾" : "补充清晰图");
            pairedVO.setWidth(1920);
            pairedVO.setHeight(1080);
            pairedVO.setHazeLevel(i == 2 ? "light" : "none");
            pairedVO.setCreateTime(LocalDateTime.now());
            pairedFiles.add(pairedVO);
        }
        vo.setPairedFiles(pairedFiles);

        // 设置数据项简要信息
        com.pei.dehaze.model.vo.DatasetItemSimpleVO datasetItem = new com.pei.dehaze.model.vo.DatasetItemSimpleVO();
        datasetItem.setId(10L);
        datasetItem.setDatasetId(5L);
        datasetItem.setName("数据项_001");
        vo.setDatasetItem(datasetItem);

        vo.setCreateTime(LocalDateTime.now());
        return vo;
    }

    /**
     * 创建模拟的BatchDeleteResultVO对象
     */
    private BatchDeleteResultVO createMockBatchDeleteResult() {
        BatchDeleteResultVO result = new BatchDeleteResultVO();
        result.setSuccessIds(Arrays.asList(1L, 2L, 3L));
        result.setFailedItems(new ArrayList<>());
        result.setSuccessCount(3);
        result.setFailedCount(0);
        return result;
    }
}
