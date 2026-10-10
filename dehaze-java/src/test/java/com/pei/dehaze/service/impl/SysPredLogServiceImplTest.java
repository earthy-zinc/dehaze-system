package com.pei.dehaze.service.impl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.core.toolkit.GlobalConfigUtils;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiModelMapper;
import com.pei.dehaze.mapper.SysAiProviderMapper;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysAiModel;
import com.pei.dehaze.model.entity.SysAiProvider;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.model.form.PredictionForm;
import com.pei.dehaze.model.vo.PredictionResultVO;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.SysDictService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.impl.file.StorageServiceFactory;
import com.pei.dehaze.service.prediction.InterceptedResult;
import com.pei.dehaze.service.prediction.PredictionInterceptorChain;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.List;
import java.util.Optional;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.util.ReflectionTestUtils;

/**
 * 图像任务模型解析与归因落库口径（设计 §9.2、§2.7）。
 *
 * <p>解析次序与 python {@code resolve_image_model} 同口径：请求显式 {@code model} > 平台默认（字典
 * {@code ai_model.default_model.image_edit}）> 拒绝；模型须启用且类型为 {@code image_edit}，
 * 其供应商亦须启用。命中模型的 {@code provider_id}/{@code actual_model} 需随任务落库（归因口径对齐 Go/Python）。
 */
@DisplayName("SysPredLogServiceImpl 图像模型解析与归因")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class SysPredLogServiceImplTest {

    private static final String DEFAULT_IMAGE_MODEL = "aecrnet";
    private static final long PROVIDER_ID = 9L;
    private static final String ORIGIN_URL = "http://cdn.test/hazy.jpg";

    @Mock
    private SysFileService sysFileService;

    @Mock
    private FileService fileService;

    @Mock
    private PredLogAsyncTask asyncTask;

    @Mock
    private PredictionInterceptorChain interceptorChain;

    @Mock
    private SysAiModelMapper aiModelMapper;

    @Mock
    private SysAiProviderMapper aiProviderMapper;

    @Mock
    private SysDictService dictService;

    @Mock
    private SysPredLogMapper predLogMapper;

    private final MeterRegistry meterRegistry = new SimpleMeterRegistry();

    private SysPredLogServiceImpl service;

    @BeforeAll
    static void initTableInfo() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        GlobalConfigUtils.setGlobalConfig(configuration, GlobalConfigUtils.defaults());
        for (Class<?> entity : List.of(SysPredLog.class, SysAiModel.class)) {
            TableInfoHelper.initTableInfo(new MapperBuilderAssistant(configuration, ""), entity);
        }
    }

    @BeforeEach
    void setUp() {
        // 工厂按 getStorageType() 建索引，须先完成桩注册
        when(fileService.getStorageType()).thenReturn("local");
        service = new SysPredLogServiceImpl(
                sysFileService,
                new StorageServiceFactory(List.of(fileService), "local"),
                asyncTask,
                interceptorChain,
                meterRegistry,
                aiModelMapper,
                aiProviderMapper,
                dictService);
        ReflectionTestUtils.setField(service, "baseMapper", predLogMapper);
        // 落库后自增主键由 mapper 回填
        when(predLogMapper.insert(any())).thenAnswer(invocation -> {
            SysPredLog entity = invocation.getArgument(0);
            entity.setId(77L);
            return 1;
        });
    }

    @Test
    @DisplayName("请求指定模型：落库 model_id/provider_id/actual_model 并异步提交该模型")
    void persistsModelAndProviderAttribution() {
        givenModel(DEFAULT_IMAGE_MODEL, 1, "image_edit");
        givenProvider(PROVIDER_ID, 1);
        givenOriginFile(5L);

        PredictionForm form = new PredictionForm();
        form.setModel(DEFAULT_IMAGE_MODEL);
        form.setFileId(5L);

        PredictionResultVO vo = service.predict(form);

        assertThat(vo.getTaskId()).isEqualTo(77L);
        assertThat(vo.getStatus()).isEqualTo(LogStatusEnum.PROCESSING.getValue());
        SysPredLog saved = capturedLog();
        assertThat(saved.getModelId()).isEqualTo(DEFAULT_IMAGE_MODEL);
        assertThat(saved.getProviderId()).isEqualTo(PROVIDER_ID);
        assertThat(saved.getActualModel()).isEqualTo(DEFAULT_IMAGE_MODEL);
        assertThat(saved.getStatus()).isEqualTo(LogStatusEnum.PROCESSING);
        assertThat(saved.getOriginUrl()).isEqualTo(ORIGIN_URL);
        verify(asyncTask).execute(77L, DEFAULT_IMAGE_MODEL, 5L, ORIGIN_URL, null);
    }

    @Test
    @DisplayName("请求未指定模型：取字典默认图像模型（ai_model.default_model.image_edit）")
    void fallsBackToPlatformDefaultModel() {
        when(dictService.getStringValue("ai_model", "default_model.image_edit", null))
                .thenReturn(DEFAULT_IMAGE_MODEL);
        givenModel(DEFAULT_IMAGE_MODEL, 1, "image_edit");
        givenProvider(PROVIDER_ID, 1);

        PredictionForm form = new PredictionForm();
        form.setImageUrl(ORIGIN_URL);

        assertThat(service.predict(form).getStatus()).isEqualTo(LogStatusEnum.PROCESSING.getValue());
        assertThat(capturedLog().getModelId()).isEqualTo(DEFAULT_IMAGE_MODEL);
    }

    @Test
    @DisplayName("请求未指定模型且字典缺项：A0500 拒绝提交")
    void rejectsWhenDefaultModelMissing() {
        PredictionForm form = new PredictionForm();
        form.setImageUrl(ORIGIN_URL);

        assertThatThrownBy(() -> service.predict(form))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("default_model.image_edit")
                .extracting(e -> ((BusinessException) e).getResultCode())
                .isEqualTo(ResultCode.BUSINESS_ERROR);
        verifyNoInteractions(asyncTask);
    }

    @Test
    @DisplayName("模型不存在/已停用/类型不符：A0400 拒绝提交")
    void rejectsUnavailableModel() {
        givenModel(DEFAULT_IMAGE_MODEL, 0, "image_edit");

        PredictionForm form = new PredictionForm();
        form.setModel(DEFAULT_IMAGE_MODEL);
        form.setImageUrl(ORIGIN_URL);

        assertThatThrownBy(() -> service.predict(form))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("图像模型不可用")
                .extracting(e -> ((BusinessException) e).getResultCode())
                .isEqualTo(ResultCode.PARAM_ERROR);
        verifyNoInteractions(asyncTask);
    }

    @Test
    @DisplayName("模型类型非 image_edit：A0400 拒绝（不得被图像链路调用）")
    void rejectsNonImageModelType() {
        givenModel("gpt-4o", 1, "chat");
        givenProvider(PROVIDER_ID, 1);

        PredictionForm form = new PredictionForm();
        form.setModel("gpt-4o");
        form.setImageUrl(ORIGIN_URL);

        assertThatThrownBy(() -> service.predict(form))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode())
                .isEqualTo(ResultCode.PARAM_ERROR);
        verifyNoInteractions(asyncTask);
    }

    @Test
    @DisplayName("供应商已停用：A0500 拒绝（对齐 python resolve_image_model）")
    void rejectsDisabledProvider() {
        givenModel(DEFAULT_IMAGE_MODEL, 1, "image_edit");
        givenProvider(PROVIDER_ID, 0);

        PredictionForm form = new PredictionForm();
        form.setModel(DEFAULT_IMAGE_MODEL);
        form.setImageUrl(ORIGIN_URL);

        assertThatThrownBy(() -> service.predict(form))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("供应商不可用")
                .extracting(e -> ((BusinessException) e).getResultCode())
                .isEqualTo(ResultCode.BUSINESS_ERROR);
        verifyNoInteractions(asyncTask);
    }

    @Test
    @DisplayName("拦截器命中（WPXNet）：直接写 completed，不提交异步推理")
    void interceptedTaskSkipsInference() {
        givenModel(DEFAULT_IMAGE_MODEL, 1, "image_edit");
        givenProvider(PROVIDER_ID, 1);
        givenOriginFile(5L);
        when(interceptorChain.intercept(any()))
                .thenReturn(Optional.of(InterceptedResult.builder()
                        .resultUrl("http://cdn.test/clean.jpg")
                        .resultMd5("clean-md5")
                        .resultFileId(88L)
                        .build()));

        PredictionForm form = new PredictionForm();
        form.setModel(DEFAULT_IMAGE_MODEL);
        form.setFileId(5L);

        PredictionResultVO vo = service.predict(form);

        assertThat(vo.getStatus()).isEqualTo(LogStatusEnum.COMPLETED.getValue());
        assertThat(vo.getResultUrl()).isEqualTo("http://cdn.test/clean.jpg");
        SysPredLog saved = capturedLog();
        assertThat(saved.getStatus()).isEqualTo(LogStatusEnum.COMPLETED);
        assertThat(saved.getProviderId()).isEqualTo(PROVIDER_ID);
        assertThat(saved.getActualModel()).isEqualTo(DEFAULT_IMAGE_MODEL);
        verify(asyncTask, never()).execute(any(), any(), any(), any(), any());
    }

    @Test
    @DisplayName("图片来源缺失（fileId 与 imageUrl 均空）：A0410 拒绝")
    void rejectsMissingImageSource() {
        givenModel(DEFAULT_IMAGE_MODEL, 1, "image_edit");
        givenProvider(PROVIDER_ID, 1);

        PredictionForm form = new PredictionForm();
        form.setModel(DEFAULT_IMAGE_MODEL);

        assertThatThrownBy(() -> service.predict(form))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode())
                .isEqualTo(ResultCode.PARAM_IS_NULL);
        verifyNoInteractions(asyncTask);
    }

    private void givenModel(String modelId, int status, String modelType) {
        SysAiModel model = new SysAiModel();
        model.setId(1L);
        model.setProviderId(PROVIDER_ID);
        model.setModelId(modelId);
        model.setModelType(modelType);
        model.setStatus(status);
        when(aiModelMapper.selectList(any())).thenReturn(List.of(model));
    }

    private void givenProvider(long providerId, int status) {
        SysAiProvider provider = new SysAiProvider();
        provider.setId(providerId);
        provider.setProviderCode("local-modal");
        provider.setStatus(status);
        when(aiProviderMapper.selectById(providerId)).thenReturn(provider);
    }

    private void givenOriginFile(long fileId) {
        SysFile file = new SysFile();
        file.setId(fileId);
        file.setStorage("local");
        file.setObjectName("dehaze/hazy.jpg");
        file.setMd5("origin-md5");
        when(sysFileService.getById(fileId)).thenReturn(file);
        when(fileService.getUrl("dehaze/hazy.jpg")).thenReturn(ORIGIN_URL);
    }

    private SysPredLog capturedLog() {
        ArgumentCaptor<SysPredLog> captor = ArgumentCaptor.forClass(SysPredLog.class);
        verify(predLogMapper).insert(captor.capture());
        return captor.getValue();
    }
}
