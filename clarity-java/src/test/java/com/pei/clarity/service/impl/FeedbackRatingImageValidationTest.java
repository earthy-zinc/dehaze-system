package com.pei.clarity.service.impl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.lenient;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.clarity.common.exception.BusinessException;
import com.pei.clarity.common.result.ResultCode;
import com.pei.clarity.mapper.SysAiModelMapper;
import com.pei.clarity.mapper.SysFeedbackReplyMapper;
import com.pei.clarity.mapper.SysPredLogMapper;
import com.pei.clarity.mapper.SysUserMapper;
import com.pei.clarity.model.form.FeedbackCreateForm;
import com.pei.clarity.model.form.RatingCreateForm;
import com.pei.clarity.mq.RabbitMQPublisher;
import com.pei.clarity.service.FileService;
import com.pei.clarity.service.MemberService;
import com.pei.clarity.service.SysDictService;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.data.redis.core.StringRedisTemplate;

/**
 * 反馈/评价图片 URL 校验单元测试。
 * <p>
 * 域名按各存储后端 baseUrl <b>前缀</b>比对（非仅 host）；数量超限/域名不合法报 A0400，
 * 扩展名不合法报 A0701「文件格式不支持」（与 Python/Go 及反馈评价后端实现文档一致）。
 */
@DisplayName("反馈/评价图片 URL 校验单元测试")
@ExtendWith(MockitoExtension.class)
class FeedbackRatingImageValidationTest {

    private static final String STORAGE_BASE_URL = "http://cdn.test:9000";

    @Mock
    private SysPredLogMapper predLogMapper;

    @Mock
    private SysAiModelMapper aiModelMapper;

    @Mock
    private SysUserMapper userMapper;

    @Mock
    private StringRedisTemplate stringRedisTemplate;

    @Mock
    private MemberService memberService;

    @Mock
    private SysDictService sysDictService;

    @Mock
    private ObjectProvider<RabbitMQPublisher> rabbitMQPublisherProvider;

    @Mock
    private SysFeedbackReplyMapper feedbackReplyMapper;

    @Mock
    private FileService fileService;

    private RatingServiceImpl ratingService;
    private FeedbackServiceImpl feedbackService;

    @BeforeEach
    void setUp() {
        lenient().when(fileService.getBaseUrl()).thenReturn(STORAGE_BASE_URL);
        List<FileService> fileServices = List.of(fileService);
        ratingService = new RatingServiceImpl(
                predLogMapper,
                aiModelMapper,
                userMapper,
                new ObjectMapper(),
                stringRedisTemplate,
                memberService,
                sysDictService,
                rabbitMQPublisherProvider,
                fileServices);
        feedbackService = new FeedbackServiceImpl(
                feedbackReplyMapper, userMapper, new ObjectMapper(), stringRedisTemplate, fileServices);
    }

    private RatingCreateForm ratingForm(String imageUrl) {
        RatingCreateForm form = new RatingCreateForm();
        form.setPredLogId(1L);
        form.setRating(5);
        form.setImageUrls(List.of(imageUrl));
        return form;
    }

    private FeedbackCreateForm feedbackForm(String imageUrl) {
        FeedbackCreateForm form = new FeedbackCreateForm();
        form.setFeedbackType("suggestion");
        form.setTitle("测试反馈标题");
        form.setContent("测试反馈内容");
        form.setImages(List.of(imageUrl));
        return form;
    }

    private void assertThrowsCode(Runnable action, ResultCode expected) {
        assertThatThrownBy(action::run)
                .isInstanceOf(BusinessException.class)
                .satisfies(
                        e -> assertThat(((BusinessException) e).getResultCode().getCode())
                                .isEqualTo(expected.getCode()));
    }

    @Test
    @DisplayName("评价截图扩展名不合法报 A0701")
    void ratingRejectsUnsupportedImageExtension() {
        assertThrowsCode(
                () -> ratingService.createRating(ratingForm(STORAGE_BASE_URL + "/img1.gif")),
                ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH);
        assertThatThrownBy(() -> ratingService.createRating(ratingForm(STORAGE_BASE_URL + "/img1.gif")))
                .isInstanceOf(BusinessException.class)
                .hasMessage(ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH.getMsg());
    }

    @Test
    @DisplayName("反馈图片扩展名不合法报 A0701")
    void feedbackRejectsUnsupportedImageExtension() {
        assertThrowsCode(
                () -> feedbackService.createFeedback(feedbackForm(STORAGE_BASE_URL + "/f1.txt")),
                ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH);
    }

    @Test
    @DisplayName("评价截图域名不在存储后端前缀内报 A0400（同 host 不同路径也不放行）")
    void ratingRejectsUrlOutsideStoragePrefix() {
        assertThrowsCode(
                () -> ratingService.createRating(ratingForm("http://127.0.0.1:9000/img1.jpg")), ResultCode.PARAM_ERROR);
        assertThrowsCode(
                () -> ratingService.createRating(ratingForm("http://cdn.test:9110/other/img1.jpg")),
                ResultCode.PARAM_ERROR);
    }

    @Test
    @DisplayName("合法域名与扩展名（含大写、带 query）通过校验，继续走后续流程")
    void ratingAcceptsValidUrlAndProceeds() {
        // 校验通过后进入预测记录查询：predLogMapper 返回 null → A0546，
        // 以此证明未在校验阶段被拦截
        assertThrowsCode(
                () -> ratingService.createRating(ratingForm(STORAGE_BASE_URL + "/img1.JPG?x=1")),
                ResultCode.PREDICTION_LOG_NOT_FOUND);
    }
}
