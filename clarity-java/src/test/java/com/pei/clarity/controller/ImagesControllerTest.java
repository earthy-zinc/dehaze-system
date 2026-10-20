package com.pei.clarity.controller;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.core.toolkit.GlobalConfigUtils;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.clarity.common.enums.LogStatusEnum;
import com.pei.clarity.common.exception.GlobalExceptionHandler;
import com.pei.clarity.model.entity.SysPredLog;
import com.pei.clarity.model.form.PredictionForm;
import com.pei.clarity.model.vo.PredictionResultVO;
import com.pei.clarity.security.util.SecurityUtils;
import com.pei.clarity.service.SysPredLogService;
import java.util.Map;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.ArgumentMatchers;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.validation.beanvalidation.LocalValidatorFactoryBean;

/**
 * 图像任务端点契约（设计 §9.2）：请求体字段名、响应字段 {@code taskId}（非 {@code logId}）、
 * 任务归属防枚举、取消幂等。
 *
 * <p>他人任务与不存在任务同口径 A0401，与 python {@code get_image_task}/{@code cancel_image_task} 一致
 * （业务异常走 400 信封，而非 200 携带错误码）。
 */
@DisplayName("ImagesController 图像任务端点")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class ImagesControllerTest {

    private static final long OWNER_ID = 2L;
    private static final long TASK_ID = 1L;

    @Mock
    private SysPredLogService predLogService;

    private MockMvc mockMvc;

    @BeforeAll
    static void initTableInfo() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        GlobalConfigUtils.setGlobalConfig(configuration, GlobalConfigUtils.defaults());
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(configuration, ""), SysPredLog.class);
    }

    @BeforeEach
    void setUp() {
        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();
        mockMvc = MockMvcBuilders.standaloneSetup(new ImagesController(predLogService))
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .build();
    }

    @Test
    @DisplayName("提交任务：请求体 model/fileId/params/recommendedBy 原样入参，响应字段名为 taskId")
    void submitBindsContractFields() throws Exception {
        when(predLogService.predict(any())).thenAnswer(invocation -> {
            PredictionResultVO vo = new PredictionResultVO();
            vo.setTaskId(TASK_ID);
            vo.setStatus(LogStatusEnum.PROCESSING.getValue());
            return vo;
        });

        String body = new ObjectMapper()
                .writeValueAsString(
                        Map.of("model", "aecrnet", "fileId", 5, "params", "{\"strength\":0.8}", "recommendedBy", 12));

        mockMvc.perform(post("/api/v1/images/edits")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.taskId").value(TASK_ID))
                .andExpect(jsonPath("$.data.status").value(1))
                .andExpect(jsonPath("$.data.logId").doesNotExist());

        ArgumentCaptor<PredictionForm> captor = ArgumentCaptor.forClass(PredictionForm.class);
        verify(predLogService).predict(captor.capture());
        PredictionForm form = captor.getValue();
        assertThat(form.getModel()).isEqualTo("aecrnet");
        assertThat(form.getFileId()).isEqualTo(5L);
        assertThat(form.getParams()).isEqualTo("{\"strength\":0.8}");
        assertThat(form.getRecommendedBy()).isEqualTo(12L);
    }

    @Test
    @DisplayName("查询本人已完成任务：返回 taskId/status/结果字段")
    void getOwnCompletedTask() throws Exception {
        when(predLogService.getById(TASK_ID)).thenReturn(predLog(LogStatusEnum.COMPLETED));

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(OWNER_ID);
            mockMvc.perform(get("/api/v1/images/tasks/{taskId}", TASK_ID))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.data.taskId").value(TASK_ID))
                    .andExpect(jsonPath("$.data.status").value(2))
                    .andExpect(jsonPath("$.data.resultUrl").value("http://cdn.test/clean.jpg"));
        }
    }

    @Test
    @DisplayName("查询他人任务：A0401（与不存在同口径防枚举）")
    void getOthersTaskRejected() throws Exception {
        when(predLogService.getById(TASK_ID)).thenReturn(predLog(LogStatusEnum.COMPLETED));

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(99L);
            mockMvc.perform(get("/api/v1/images/tasks/{taskId}", TASK_ID))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.code").value("A0401"));
        }
    }

    @Test
    @DisplayName("查询不存在的任务：A0401")
    void getMissingTaskRejected() throws Exception {
        when(predLogService.getById(TASK_ID)).thenReturn(null);

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(OWNER_ID);
            mockMvc.perform(get("/api/v1/images/tasks/{taskId}", TASK_ID))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.code").value("A0401"));
        }
    }

    @Test
    @DisplayName("取消处理中任务：置已取消(4)")
    void cancelProcessingTask() throws Exception {
        when(predLogService.getById(TASK_ID)).thenReturn(predLog(LogStatusEnum.PROCESSING));
        when(predLogService.update(ArgumentMatchers.<Wrapper<SysPredLog>>any())).thenReturn(true);

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(OWNER_ID);
            mockMvc.perform(post("/api/v1/images/tasks/{taskId}/cancel", TASK_ID))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.data.taskId").value(TASK_ID))
                    .andExpect(jsonPath("$.data.status").value(4));
        }
        verify(predLogService).update(ArgumentMatchers.<Wrapper<SysPredLog>>any());
    }

    @Test
    @DisplayName("取消幂等：已完成任务返回当前状态且不写库")
    void cancelCompletedTaskIsIdempotent() throws Exception {
        when(predLogService.getById(TASK_ID)).thenReturn(predLog(LogStatusEnum.COMPLETED));

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(OWNER_ID);
            mockMvc.perform(post("/api/v1/images/tasks/{taskId}/cancel", TASK_ID))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.data.status").value(2));
        }
        verify(predLogService, never()).update(ArgumentMatchers.<Wrapper<SysPredLog>>any());
    }

    @Test
    @DisplayName("并发取消：前置条件不满足时保持原状态")
    void cancelKeepsStatusWhenConditionalUpdateMisses() throws Exception {
        when(predLogService.getById(TASK_ID)).thenReturn(predLog(LogStatusEnum.PROCESSING));
        when(predLogService.update(ArgumentMatchers.<Wrapper<SysPredLog>>any())).thenReturn(false);

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(OWNER_ID);
            mockMvc.perform(post("/api/v1/images/tasks/{taskId}/cancel", TASK_ID))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.data.status").value(1));
        }
    }

    @Test
    @DisplayName("取消他人任务：A0401")
    void cancelOthersTaskRejected() throws Exception {
        when(predLogService.getById(TASK_ID)).thenReturn(predLog(LogStatusEnum.PROCESSING));

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(99L);
            mockMvc.perform(post("/api/v1/images/tasks/{taskId}/cancel", TASK_ID))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.code").value("A0401"));
        }
        verify(predLogService, never()).update(ArgumentMatchers.<Wrapper<SysPredLog>>any());
    }

    private SysPredLog predLog(LogStatusEnum status) {
        SysPredLog log = new SysPredLog();
        log.setId(TASK_ID);
        log.setCreateBy(OWNER_ID);
        log.setModelId("aecrnet");
        log.setStatus(status);
        log.setPredUrl("http://cdn.test/clean.jpg");
        log.setPredMd5("clean-md5");
        log.setOriginMd5("origin-md5");
        log.setTime(1);
        return log;
    }
}
