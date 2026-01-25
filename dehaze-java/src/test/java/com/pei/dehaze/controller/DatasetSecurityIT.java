package com.pei.dehaze.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.model.form.DatasetAddForm;
import com.pei.dehaze.config.TestConfig;
import com.pei.dehaze.model.vo.DatasetVO;
import com.pei.dehaze.service.DatasetOperationService;
import com.pei.dehaze.service.SysDatasetService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.MediaType;
import org.springframework.security.test.context.support.WithAnonymousUser;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.web.servlet.MockMvc;

import java.util.Collections;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * 数据集控制器安全测试
 * 测试目的：验证数据集API的权限控制是否正确
 * 测试场景：
 * 1. 未登录用户访问受保护接口
 * 2. 已登录用户正常访问接口
 * 3. 不同角色的权限控制
 */
@WebMvcTest(SysDatasetController.class)
@Import(TestConfig.class)
@DisplayName("数据集安全测试")
class DatasetSecurityIT {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private SysDatasetService datasetService;

    @MockBean
    private DatasetOperationService datasetOperationService;

    @MockBean
    private RedisTemplate<String, Object> redisTemplate;

    private DatasetAddForm validAddForm;

    @BeforeEach
    void setUp() {
        validAddForm = new DatasetAddForm();
        validAddForm.setName("测试数据集");
        validAddForm.setType("image");
        validAddForm.setDescription("测试描述");
        validAddForm.setParentId(0L);
    }

    /**
     * 未认证用户测试
     * 验证：未登录用户访问受保护接口应返回401
     */
    @Nested
    @DisplayName("未认证用户测试")
    class UnauthenticatedUserTests {

        /**
         * 测试未登录用户访问数据集列表
         * 期望：返回401未授权
         */
        @Test
        @WithAnonymousUser
        @DisplayName("未登录用户访问数据集列表应返回401")
        void listDatasets_WithAnonymousUser_ShouldReturn401() throws Exception {
            mockMvc.perform(get("/api/v1/datasets"))
                    .andExpect(status().isUnauthorized());
        }

        /**
         * 测试未登录用户获取数据集详情
         * 期望：返回401未授权
         */
        @Test
        @WithAnonymousUser
        @DisplayName("未登录用户获取数据集详情应返回401")
        void getDatasetById_WithAnonymousUser_ShouldReturn401() throws Exception {
            mockMvc.perform(get("/api/v1/datasets/1"))
                    .andExpect(status().isUnauthorized());
        }

        /**
         * 测试未登录用户新增数据集
         * 期望：返回401未授权
         */
        @Test
        @WithAnonymousUser
        @DisplayName("未登录用户新增数据集应返回401")
        void addDataset_WithAnonymousUser_ShouldReturn401() throws Exception {
            mockMvc.perform(post("/api/v1/datasets")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(objectMapper.writeValueAsString(validAddForm)))
                    .andExpect(status().isUnauthorized());
        }

        /**
         * 测试未登录用户删除数据集
         * 期望：返回401未授权
         */
        @Test
        @WithAnonymousUser
        @DisplayName("未登录用户删除数据集应返回401")
        void deleteDataset_WithAnonymousUser_ShouldReturn401() throws Exception {
            mockMvc.perform(delete("/api/v1/datasets/1"))
                    .andExpect(status().isUnauthorized());
        }
    }

    /**
     * 已认证用户测试
     * 验证：已登录用户可以正常访问接口
     */
    @Nested
    @DisplayName("已认证用户测试")
    class AuthenticatedUserTests {

        /**
         * 测试已登录用户访问数据集列表
         * 期望：返回200成功
         */
        @Test
        @WithMockUser(username = "testuser", roles = {"USER"})
        @DisplayName("已登录用户访问数据集列表应返回200")
        void listDatasets_WithAuthenticatedUser_ShouldReturn200() throws Exception {
            when(datasetService.getList(any())).thenReturn(Collections.emptyList());

            mockMvc.perform(get("/api/v1/datasets"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.code").value(ResultCode.SUCCESS.getCode()));
        }

        /**
         * 测试已登录用户获取数据集详情
         * 期望：返回200成功
         */
        @Test
        @WithMockUser(username = "testuser", roles = {"USER"})
        @DisplayName("已登录用户获取数据集详情应返回200")
        void getDatasetById_WithAuthenticatedUser_ShouldReturn200() throws Exception {
            DatasetVO mockVO = new DatasetVO();
            mockVO.setId(1L);
            mockVO.setName("测试数据集");
            when(datasetService.getDatasetById(1L)).thenReturn(mockVO);

            mockMvc.perform(get("/api/v1/datasets/1"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.code").value(ResultCode.SUCCESS.getCode()));
        }

        /**
         * 测试已登录用户新增数据集
         * 期望：返回200成功
         */
        @Test
        @WithMockUser(username = "testuser", roles = {"USER"})
        @DisplayName("已登录用户新增数据集应返回200")
        void addDataset_WithAuthenticatedUser_ShouldReturn200() throws Exception {
            DatasetVO mockVO = new DatasetVO();
            mockVO.setId(1L);
            mockVO.setName("测试数据集");
            when(datasetService.addDataset(any())).thenReturn(mockVO);

            mockMvc.perform(post("/api/v1/datasets")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content(objectMapper.writeValueAsString(validAddForm)))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.code").value(ResultCode.SUCCESS.getCode()));
        }

        /**
         * 测试已登录用户删除数据集
         * 期望：返回200成功
         */
        @Test
        @WithMockUser(username = "testuser", roles = {"USER"})
        @DisplayName("已登录用户删除数据集应返回200")
        void deleteDataset_WithAuthenticatedUser_ShouldReturn200() throws Exception {
            mockMvc.perform(delete("/api/v1/datasets/1"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.code").value(ResultCode.SUCCESS.getCode()));
        }
    }

    /**
     * 管理员角色测试
     * 验证：管理员可以访问所有接口
     */
    @Nested
    @DisplayName("管理员角色测试")
    class AdminRoleTests {

        /**
         * 测试管理员访问数据集列表
         * 期望：返回200成功
         */
        @Test
        @WithMockUser(username = "admin", roles = {"ADMIN"})
        @DisplayName("管理员访问数据集列表应返回200")
        void listDatasets_WithAdminRole_ShouldReturn200() throws Exception {
            when(datasetService.getList(any())).thenReturn(Collections.emptyList());

            mockMvc.perform(get("/api/v1/datasets"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.code").value(ResultCode.SUCCESS.getCode()));
        }

        /**
         * 测试管理员批量删除数据集
         * 期望：返回200成功
         */
        @Test
        @WithMockUser(username = "admin", roles = {"ADMIN"})
        @DisplayName("管理员批量删除数据集应返回200")
        void batchDeleteDatasets_WithAdminRole_ShouldReturn200() throws Exception {
            when(datasetOperationService.batchDeleteDatasets(any())).thenReturn(null);

            mockMvc.perform(delete("/api/v1/datasets/batch")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content("{\"ids\":[1,2,3]}"))
                    .andExpect(status().isOk());
        }
    }
}
