package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.service.SysUserService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.web.context.WebApplicationContext;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.security.test.web.servlet.setup.SecurityMockMvcConfigurers.springSecurity;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;

@SpringBootTest
@AutoConfigureMockMvc(addFilters = false)
class SysUserControllerTest {
    @Autowired
    private WebApplicationContext context;

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private SysUserService userService;

    @BeforeEach
    void setup() {
        // 显式初始化MockMvc
        this.mockMvc = MockMvcBuilders
                .webAppContextSetup(this.context)
                .apply(springSecurity())  // 启用安全测试支持
                .build();
    }

    // 用户分页查询测试
    @Test
    @WithMockUser(roles = "ADMIN")
    void testUserPagination_Success() throws Exception {
        // 模拟分页数据
        when(userService.listPagedUsers(any())).thenReturn(new Page<>(1, 10));

        mockMvc.perform(get("/api/v1/users/page")
                        .param("pageNum", "1")
                        .param("pageSize", "10"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    // 异常权限测试
    @Test
    @WithMockUser(roles = "USER")
    void testDeleteUser_Forbidden() throws Exception {
        mockMvc.perform(delete("/api/v1/users/1"))
                .andExpect(status().is4xxClientError());
    }

    // 用户导入异常测试
    @Test
    void testUserImport_InvalidFile() throws Exception {
        MockMultipartFile invalidFile = new MockMultipartFile(
                "file", "test.txt", "text/plain", "invalid content".getBytes());

        mockMvc.perform(multipart("/api/v1/users/_import")
                        .file(invalidFile).param("deptId", "1"))
                .andExpect(status().is4xxClientError())
                .andExpect(jsonPath("$.code").value("A0230"));
    }
}
