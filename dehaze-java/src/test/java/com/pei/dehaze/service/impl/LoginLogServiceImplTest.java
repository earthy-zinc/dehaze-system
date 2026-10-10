package com.pei.dehaze.service.impl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.mongodb.MongoClientSettings;
import com.pei.dehaze.model.entity.LoginLog;
import com.pei.dehaze.model.query.LoginLogQuery;
import com.pei.dehaze.model.vo.LoginLogVO;
import com.pei.dehaze.repository.LoginLogRepository;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import org.bson.BsonDocument;
import org.bson.BsonType;
import org.bson.BsonValue;
import org.bson.Document;
import org.bson.codecs.configuration.CodecRegistry;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Query;

/**
 * LoginLogServiceImpl 单元测试。
 *
 * <p>锁定三条易回归的事实：① 条件/排序键必须是与 python 共用的 snake_case
 * （{@code user_id/device_type/create_time}）；② 起止时间同时传入时不得因同字段双 and
 * 触发 {@code InvalidMongoDbApiUsageException}（取 Query 条件对象即会暴露）；
 * ③ {@code user_id} 必须走 int32/int64 双类型 {@code $in}——java 用 {@code Long}（int64）匹配时
 * 命不中 python 历史记录（pymongo 对 &lt; 2^31 的整数写 int32），且静默返回空列表。
 */
@DisplayName("LoginLogServiceImpl 单元测试")
@ExtendWith(MockitoExtension.class)
class LoginLogServiceImplTest {

    private static final CodecRegistry CODEC_REGISTRY = MongoClientSettings.getDefaultCodecRegistry();

    @Mock
    private LoginLogRepository loginLogRepository;

    @Mock
    private MongoTemplate mongoTemplate;

    @InjectMocks
    private LoginLogServiceImpl service;

    @Test
    @DisplayName("分页查询：snake_case 条件键 + 起止时间同传不崩 + create_time 倒序")
    void pageLoginLogs_usesSnakeCaseKeysAndSingleTimeChain() {
        when(mongoTemplate.count(any(Query.class), eq(LoginLog.class))).thenReturn(3L);
        when(mongoTemplate.find(any(Query.class), eq(LoginLog.class))).thenReturn(List.of());
        LoginLogQuery query = new LoginLogQuery();
        query.setPageNum(2);
        query.setPageSize(10);
        query.setUsername("admin");
        query.setIp("10.0.0.1");
        query.setStatus(1);
        query.setDeviceType("web");
        query.setStartTime("2026-09-01 00:00:00");
        query.setEndTime("2026-09-30 23:59:59");

        IPage<LoginLogVO> page = service.pageLoginLogs(query, 7L);

        assertThat(page.getTotal()).isEqualTo(3L);
        ArgumentCaptor<Query> captor = ArgumentCaptor.forClass(Query.class);
        verify(mongoTemplate).find(captor.capture(), eq(LoginLog.class));
        Query mongoQuery = captor.getValue();
        Document filter = mongoQuery.getQueryObject();
        assertThat(filter.get("username")).isEqualTo("admin");
        assertThat(filter.get("ip")).isEqualTo("10.0.0.1");
        assertThat(filter.get("status")).isEqualTo(1);
        assertThat(filter.get("device_type")).isEqualTo("web");
        // 值 + BSON 类型双断言：Long（int64 口径）在前，Integer（python 历史 int32）在后
        assertThat(candidatesOf(filter, "user_id")).containsExactly(7L, 7);
        assertThat(bsonTypesOfCandidates(filter, "user_id"))
                .containsExactlyInAnyOrder(BsonType.INT64, BsonType.INT32);
        assertThat(((Document) filter.get("create_time")).keySet()).containsExactlyInAnyOrder("$gte", "$lte");
        assertThat(mongoQuery.getSortObject()).isEqualTo(new Document("create_time", -1));
        assertThat(mongoQuery.getSkip()).isEqualTo(10L);
        assertThat(mongoQuery.getLimit()).isEqualTo(10);
    }

    @Test
    @DisplayName("分页查询：int32 历史记录可读（user_id 双类型 $in），超出 int32 值域时只留 int64 候选")
    void pageLoginLogs_userIdMatchesLegacyInt32Records() {
        when(mongoTemplate.count(any(Query.class), eq(LoginLog.class))).thenReturn(0L);
        when(mongoTemplate.find(any(Query.class), eq(LoginLog.class))).thenReturn(List.of());

        service.pageLoginLogs(new LoginLogQuery(), 790001L);
        // 越界值：python 对 ≥ 2^31 的整数同样写 int64，故只应有 int64 候选（intValue() 会溢出）
        service.pageLoginLogs(new LoginLogQuery(), (long) Integer.MAX_VALUE + 1);

        ArgumentCaptor<Query> captor = ArgumentCaptor.forClass(Query.class);
        verify(mongoTemplate, times(2)).find(captor.capture(), eq(LoginLog.class));
        List<Query> queries = captor.getAllValues();

        Document int32Range = queries.get(0).getQueryObject();
        assertThat(candidatesOf(int32Range, "user_id")).containsExactly(790001L, 790001);
        assertThat(bsonTypesOfCandidates(int32Range, "user_id"))
                .containsExactlyInAnyOrder(BsonType.INT64, BsonType.INT32);

        Document outOfInt32Range = queries.get(1).getQueryObject();
        assertThat(candidatesOf(outOfInt32Range, "user_id"))
                .containsExactly((long) Integer.MAX_VALUE + 1);
        assertThat(bsonTypesOfCandidates(outOfInt32Range, "user_id")).containsExactly(BsonType.INT64);
    }

    @Test
    @DisplayName("分页查询：无时间筛选时不生成 create_time 条件，视图字段与时间格式正确")
    void pageLoginLogs_omitsTimeFilterAndMapsView() {
        LoginLog log = new LoginLog();
        log.setId("66f0c2b8e1a2b3c4d5e6f7a8");
        log.setUserId(7L);
        log.setUsername("admin");
        log.setIp("10.0.0.1");
        log.setDeviceType(" ");
        log.setStatus(1);
        log.setCreateTime(LocalDateTime.of(2026, 9, 17, 10, 30, 0));
        when(mongoTemplate.count(any(Query.class), eq(LoginLog.class))).thenReturn(1L);
        when(mongoTemplate.find(any(Query.class), eq(LoginLog.class))).thenReturn(List.of(log));

        IPage<LoginLogVO> page = service.pageLoginLogs(new LoginLogQuery(), null);

        ArgumentCaptor<Query> captor = ArgumentCaptor.forClass(Query.class);
        verify(mongoTemplate).count(captor.capture(), eq(LoginLog.class));
        assertThat(captor.getValue().getQueryObject()).doesNotContainKey("create_time");
        assertThat(captor.getValue().getQueryObject()).doesNotContainKey("user_id");
        LoginLogVO vo = page.getRecords().get(0);
        assertThat(vo.getId()).isEqualTo("66f0c2b8e1a2b3c4d5e6f7a8");
        assertThat(vo.getDeviceType()).isEqualTo("web");
        assertThat(vo.getLoginTime()).isEqualTo("2026-09-17 10:30:00");
    }

    @Test
    @DisplayName("记录登录日志：落库实体字段完整（键名由实体 @Field 统一映射）")
    void recordLogin_savesEntity() {
        service.recordLogin(7L, "admin", "10.0.0.1", 1, "登录成功", "Chrome", "Linux", "web");

        ArgumentCaptor<LoginLog> captor = ArgumentCaptor.forClass(LoginLog.class);
        verify(loginLogRepository).save(captor.capture());
        LoginLog saved = captor.getValue();
        assertThat(saved.getUserId()).isEqualTo(7L);
        assertThat(saved.getDeviceType()).isEqualTo("web");
        assertThat(saved.getStatus()).isEqualTo(1);
        assertThat(saved.getCreateTime()).isNotNull();
    }

    /** 过滤条件中某字段 `$in` 的候选值（证明双类型候选的取值）。 */
    private static List<Object> candidatesOf(Document filter, String field) {
        return ((Document) filter.get(field)).getList("$in", Object.class);
    }

    /** 过滤条件中 `$in` 候选值的 BSON 类型（真编码：Document → BsonDocument）。 */
    private static List<BsonType> bsonTypesOfCandidates(Document filter, String field) {
        BsonDocument encoded = filter.toBsonDocument(BsonDocument.class, CODEC_REGISTRY);
        List<BsonType> types = new ArrayList<>();
        for (BsonValue value : encoded.getDocument(field).getArray("$in")) {
            types.add(value.getBsonType());
        }
        return types;
    }
}
