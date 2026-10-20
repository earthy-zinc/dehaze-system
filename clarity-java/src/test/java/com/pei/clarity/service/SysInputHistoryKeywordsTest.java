package com.pei.clarity.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.AbstractWrapper;
import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.clarity.mapper.SysInputHistoryMapper;
import com.pei.clarity.mapper.SysMemberMapper;
import com.pei.clarity.model.entity.SysInputHistory;
import com.pei.clarity.model.query.HistoryQuery;
import com.pei.clarity.security.model.SysUserDetails;
import com.pei.clarity.service.impl.SysInputHistoryServiceImpl;
import java.util.List;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.AfterEach;
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
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.test.util.ReflectionTestUtils;

/**
 * 历史记录 keywords 过滤：model_name / original_image_url 的 OR 模糊匹配，
 * 与 python(escape_like)、go(escapeLike) 同口径，且 LIKE 通配符被转义。
 *
 * <p>纯单测无 SQL 执行，以查询片段与参数断言口径。
 */
@DisplayName("SysInputHistoryService keywords 过滤")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class SysInputHistoryKeywordsTest {

    private static final Long USER_ID = 7L;

    @Mock
    private SysInputHistoryMapper historyMapper;

    @Mock
    private SysMemberMapper memberMapper;

    @Mock
    private MemberBenefitService memberBenefitService;

    private SysInputHistoryServiceImpl service;

    @BeforeAll
    static void initMybatisPlusLambdaCache() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        MapperBuilderAssistant assistant = new MapperBuilderAssistant(configuration, "input-history-keywords-test");
        assistant.setCurrentNamespace("com.pei.clarity.mapper.SysInputHistoryMapper");
        TableInfoHelper.initTableInfo(assistant, SysInputHistory.class);
    }

    @BeforeEach
    void setUp() {
        service = new SysInputHistoryServiceImpl(memberMapper, memberBenefitService);
        ReflectionTestUtils.setField(service, "baseMapper", historyMapper);
        SysUserDetails principal = new SysUserDetails();
        principal.setUserId(USER_ID);
        SecurityContextHolder.getContext()
                .setAuthentication(new UsernamePasswordAuthenticationToken(principal, null, List.of()));
        when(historyMapper.selectPage(any(Page.class), any(Wrapper.class))).thenReturn(new Page<>(1L, 10L, 0L));
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    @Test
    @DisplayName("keywords 参与 model_name / original_image_url 的 OR 模糊匹配")
    void keywordsBuildsOrLikeCondition() {
        service.getHistoryPage(queryWithKeywords("去雾"));

        assertThat(capturedWrapper().getSqlSegment())
                .contains("model_name LIKE")
                .contains("original_image_url LIKE")
                .contains("OR");
    }

    @Test
    @DisplayName("OR 条件被括号包裹，不破坏 user_id 归属过滤")
    void keywordsConditionIsGrouped() {
        service.getHistoryPage(queryWithKeywords("去雾"));

        assertThat(capturedWrapper().getSqlSegment()).startsWith("(user_id =").contains("AND (model_name LIKE");
    }

    @Test
    @DisplayName("关键词中的 %/_ 被转义，不当作通配符")
    void keywordsEscapesLikeWildcards() {
        service.getHistoryPage(queryWithKeywords("a_b%"));

        AbstractWrapper wrapper = capturedWrapper();
        wrapper.getTargetSql(); // 触发嵌套条件把参数回填到外层 wrapper
        // like() 自带两侧 % 包裹，故入参是转义后的关键词本身
        assertThat(wrapper.getParamNameValuePairs().values()).contains("%a\\_b\\%%");
    }

    @Test
    @DisplayName("keywords 为空白时不追加任何 LIKE 条件")
    void blankKeywordsAddsNoLikeCondition() {
        service.getHistoryPage(queryWithKeywords("   "));

        assertThat(capturedWrapper().getSqlSegment()).doesNotContain("LIKE");
    }

    private HistoryQuery queryWithKeywords(String keywords) {
        HistoryQuery query = new HistoryQuery();
        query.setKeywords(keywords);
        return query;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private AbstractWrapper capturedWrapper() {
        ArgumentCaptor<Wrapper<SysInputHistory>> captor = ArgumentCaptor.forClass((Class) Wrapper.class);
        verify(historyMapper).selectPage(any(Page.class), captor.capture());
        return (AbstractWrapper) captor.getValue();
    }
}
