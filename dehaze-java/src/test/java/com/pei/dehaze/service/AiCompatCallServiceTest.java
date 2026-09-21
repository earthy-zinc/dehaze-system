package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.model.entity.AiApiCallLog;
import com.pei.dehaze.model.query.AiCompatCallQuery;
import com.pei.dehaze.model.vo.AiCompatCallVO;
import com.pei.dehaze.security.util.SecurityUtils;
import org.bson.Document;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Query;

import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDateTime;
import java.util.Date;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * AiCompatCallService 单元测试。
 *
 * <p>锁定与 dehaze-python / dehaze-go 一致的审计查询口径：user_id 强制本人过滤、
 * 时间非法按无过滤、page/size 分页（create_time 倒序）、UTC 时刻解析与展示、
 * Mongo snake_case 文档键到 camelCase 视图的映射。
 */
@DisplayName("AiCompatCallService 单元测试")
@ExtendWith(MockitoExtension.class)
class AiCompatCallServiceTest {

    @Mock
    private MongoTemplate mongoTemplate;

    @InjectMocks
    private AiCompatCallService service;

    @Test
    @DisplayName("查询：强制本人 user_id 过滤 + page/size 分页 + create_time 倒序")
    void listCalls_filtersByCurrentUserAndPaging() {
        when(mongoTemplate.count(any(Query.class), eq(AiApiCallLog.class))).thenReturn(42L);
        when(mongoTemplate.find(any(Query.class), eq(AiApiCallLog.class))).thenReturn(List.of());
        AiCompatCallQuery query = new AiCompatCallQuery();
        query.setPage(2);
        query.setSize(20);

        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(7L);

            IPage<AiCompatCallVO> page = service.listCalls(query);

            assertThat(page.getTotal()).isEqualTo(42L);
            assertThat(page.getCurrent()).isEqualTo(2);
        }

        ArgumentCaptor<Query> captor = ArgumentCaptor.forClass(Query.class);
        verify(mongoTemplate).find(captor.capture(), eq(AiApiCallLog.class));
        Query mongoQuery = captor.getValue();
        assertThat(mongoQuery.getSkip()).isEqualTo(20L);
        assertThat(mongoQuery.getLimit()).isEqualTo(20);
        assertThat(mongoQuery.getSortObject()).isEqualTo(new Document("create_time", -1));
        assertThat(mongoQuery.getQueryObject().get("user_id")).isEqualTo(7L);
    }

    @Test
    @DisplayName("查询：keyId/model/时间范围过滤，时间按 UTC 时刻解析（与 python 无时区入参口径一致）")
    void listCalls_appliesFiltersWithUtcTime() {
        when(mongoTemplate.count(any(Query.class), eq(AiApiCallLog.class))).thenReturn(1L);
        when(mongoTemplate.find(any(Query.class), eq(AiApiCallLog.class))).thenReturn(List.of());
        AiCompatCallQuery query = new AiCompatCallQuery();
        query.setKeyId(3L);
        query.setModel("gpt-4o");
        query.setStartTime("2026-09-17 00:00:00");
        query.setEndTime("2026-09-17");

        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(7L);
            service.listCalls(query);
        }

        ArgumentCaptor<Query> captor = ArgumentCaptor.forClass(Query.class);
        verify(mongoTemplate).count(captor.capture(), eq(AiApiCallLog.class));
        Document filter = captor.getValue().getQueryObject();
        assertThat(filter.get("user_id")).isEqualTo(7L);
        assertThat(filter.get("key_id")).isEqualTo(3L);
        assertThat(filter.get("model")).isEqualTo("gpt-4o");
        Document timeRange = (Document) filter.get("create_time");
        assertThat(epochMillis(timeRange.get("$gte"))).isEqualTo(Instant.parse("2026-09-17T00:00:00Z").toEpochMilli());
        assertThat(epochMillis(timeRange.get("$lte"))).isEqualTo(Instant.parse("2026-09-17T00:00:00Z").toEpochMilli());
    }

    @Test
    @DisplayName("查询：时间格式非法/为空按无过滤处理（不抛错，与 python 兼容审计一致）")
    void listCalls_ignoresInvalidTimeFormat() {
        when(mongoTemplate.count(any(Query.class), eq(AiApiCallLog.class))).thenReturn(0L);
        when(mongoTemplate.find(any(Query.class), eq(AiApiCallLog.class))).thenReturn(List.of());
        AiCompatCallQuery query = new AiCompatCallQuery();
        query.setStartTime("2026/09/17");
        query.setEndTime("  ");

        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(7L);
            service.listCalls(query);
        }

        ArgumentCaptor<Query> captor = ArgumentCaptor.forClass(Query.class);
        verify(mongoTemplate).count(captor.capture(), eq(AiApiCallLog.class));
        assertThat(captor.getValue().getQueryObject()).doesNotContainKey("create_time");
    }

    @Test
    @DisplayName("查询：ISO 格式时间可解析，文档字段按 snake_case 映射到 camelCase 视图")
    void listCalls_mapsDocumentFields() {
        AiApiCallLog log = new AiApiCallLog();
        log.setId("66f0c2b8e1a2b3c4d5e6f7a8");
        log.setKeyId(3L);
        log.setKeyPrefix("dhak_abc");
        log.setConversationId(5L);
        log.setModel("gpt-4o");
        log.setEndpoint("chat/completions");
        log.setProtocol("openai");
        log.setIsStream(true);
        log.setInputTokens(120);
        log.setOutputTokens(80);
        log.setCredits(new BigDecimal("1.5"));
        log.setStatusCode(200);
        log.setDurationMs(1500);
        log.setClientIp("10.0.0.1");
        log.setRequestId("req-1");
        log.setCreateTime(Instant.parse("2026-09-17T10:00:00Z"));
        when(mongoTemplate.count(any(Query.class), eq(AiApiCallLog.class))).thenReturn(1L);
        when(mongoTemplate.find(any(Query.class), eq(AiApiCallLog.class))).thenReturn(List.of(log));
        AiCompatCallQuery query = new AiCompatCallQuery();
        query.setStartTime("2026-09-17T00:00:00");

        IPage<AiCompatCallVO> page;
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(7L);
            page = service.listCalls(query);
        }

        AiCompatCallVO vo = page.getRecords().get(0);
        assertThat(vo.getId()).isEqualTo("66f0c2b8e1a2b3c4d5e6f7a8");
        assertThat(vo.getKeyPrefix()).isEqualTo("dhak_abc");
        assertThat(vo.getIsStream()).isTrue();
        assertThat(vo.getCredits()).isEqualByComparingTo("1.5");
        assertThat(vo.getStatusCode()).isEqualTo(200);
        assertThat(vo.getCreateTime()).isEqualTo(LocalDateTime.of(2026, 9, 17, 10, 0));
        assertThat(vo.getErrorMsg()).isNull();
    }

    private static long epochMillis(Object value) {
        if (value instanceof Instant instant) {
            return instant.toEpochMilli();
        }
        if (value instanceof Date date) {
            return date.getTime();
        }
        throw new AssertionError("非时间类型: " + value);
    }
}
