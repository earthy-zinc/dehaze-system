package com.pei.dehaze.service.impl;

import cn.hutool.json.JSONObject;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysRecommendationRuleMapper;
import com.pei.dehaze.model.form.AnalyzeForm;
import com.pei.dehaze.model.vo.ImageFeatureAnalysisVO;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.client.PythonAlgorithmClient;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.when;

/**
 * RecommendationServiceImpl 单元测试
 * <p>
 * 验证图像特征分析已接入 Python 真实图像特征分析服务：
 * 正常调用返回真实特征映射、Python 不可用时不降级为伪特征、入参格式校验。
 */
@DisplayName("RecommendationServiceImpl 单元测试")
@ExtendWith(MockitoExtension.class)
class RecommendationServiceImplTest {

    @Mock
    private SysRecommendationRuleMapper ruleMapper;
    @Mock
    private SysAlgorithmService sysAlgorithmService;
    @Mock
    private PythonAlgorithmClient pythonAlgorithmClient;

    private RecommendationServiceImpl service;

    @BeforeEach
    void setUp() {
        service = new RecommendationServiceImpl(ruleMapper, sysAlgorithmService, pythonAlgorithmClient);
    }

    private JSONObject pythonFeatureData() {
        JSONObject cd = new JSONObject();
        cd.set("temperature", 5200.0);
        cd.set("saturation", 0.62);
        JSONObject data = new JSONObject();
        data.set("imageMd5", "d41d8cd98f00b204e9800998ecf8427e");
        data.set("hazeLevel", "moderate");
        data.set("hazeConfidence", 0.87);
        data.set("sceneType", "urban");
        data.set("sceneConfidence", 0.91);
        data.set("lighting", "normal");
        data.set("complexity", 0.45);
        data.set("colorDistribution", cd);
        data.set("resolution", "hd");
        data.set("noiseLevel", "medium");
        return data;
    }

    @Test
    @DisplayName("analyze - 调用 Python 服务返回真实特征并正确映射")
    void analyze_realFeatureResponse_returnsMappedVO() {
        AnalyzeForm form = new AnalyzeForm();
        form.setImageUrl("http://example.com/image.jpg");
        when(pythonAlgorithmClient.analyzeImage("http://example.com/image.jpg"))
                .thenReturn(pythonFeatureData());

        ImageFeatureAnalysisVO vo = service.analyze(form);

        assertThat(vo.getImageMd5()).isEqualTo("d41d8cd98f00b204e9800998ecf8427e");
        assertThat(vo.getHazeLevel()).isEqualTo("moderate");
        assertThat(vo.getHazeConfidence()).isEqualTo(0.87);
        assertThat(vo.getSceneType()).isEqualTo("urban");
        assertThat(vo.getSceneConfidence()).isEqualTo(0.91);
        assertThat(vo.getLighting()).isEqualTo("normal");
        assertThat(vo.getComplexity()).isEqualTo(0.45);
        assertThat(vo.getColorDistribution()).isNotNull();
        assertThat(vo.getColorDistribution().getTemperature()).isEqualTo(5200.0);
        assertThat(vo.getColorDistribution().getSaturation()).isEqualTo(0.62);
        assertThat(vo.getResolution()).isEqualTo("hd");
        assertThat(vo.getNoiseLevel()).isEqualTo("medium");
    }

    @Test
    @DisplayName("analyze - Python 服务不可用时抛出业务异常，不降级为伪特征")
    void analyze_pythonUnavailable_throwsBusinessException() {
        AnalyzeForm form = new AnalyzeForm();
        form.setImageUrl("http://example.com/image.jpg");
        when(pythonAlgorithmClient.analyzeImage("http://example.com/image.jpg"))
                .thenThrow(new BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "图像特征分析服务不可用"));

        assertThatThrownBy(() -> service.analyze(form))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("图像特征分析服务不可用");
    }

    @Test
    @DisplayName("analyze - imageId 方式暂不支持")
    void analyze_imageIdNotSupported_throws() {
        AnalyzeForm form = new AnalyzeForm();
        form.setImageId(1L);

        assertThatThrownBy(() -> service.analyze(form))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("imageId方式暂不支持");
    }

    @Test
    @DisplayName("analyze - imageId 和 imageUrl 均未提供抛出参数错误")
    void analyze_missingImageUrl_throws() {
        AnalyzeForm form = new AnalyzeForm();

        assertThatThrownBy(() -> service.analyze(form))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("imageId和imageUrl至少提供一个");
    }

    @Test
    @DisplayName("analyze - 非图片格式抛出文件类型不匹配")
    void analyze_invalidImageFormat_throws() {
        AnalyzeForm form = new AnalyzeForm();
        form.setImageUrl("http://example.com/file.pdf");

        assertThatThrownBy(() -> service.analyze(form))
                .isInstanceOf(BusinessException.class);
    }
}
