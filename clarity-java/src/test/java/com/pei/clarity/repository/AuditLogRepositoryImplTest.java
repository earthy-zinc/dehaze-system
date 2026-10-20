package com.pei.clarity.repository;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.mongodb.MongoClientSettings;
import com.pei.clarity.model.entity.AuditLog;
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
 * {@link AuditLogRepositoryImpl} 单元测试（无需 Mongo：断言构造出的 Query 与 BSON 编码）。
 *
 * <p>钉住的事实：{@code audit_log} 的数值归属字段必须按 **int32/int64 双类型** 匹配
 * ——java/go 写 int64、python(pymongo) 对 &lt; 2^31 的整数写 int32，BSON 数值类型不相等，
 * 单类型条件会静默漏读历史记录。java 侧无 Mongo 测试基建（无 embedded Mongo），故用
 * Mockito 捕获 Query + 驱动默认 codec 做 BSON 层类型断言（Long→int64、Integer→int32）。
 */
@DisplayName("AuditLogRepositoryImpl 跨端双类型查询")
@ExtendWith(MockitoExtension.class)
class AuditLogRepositoryImplTest {

    private static final CodecRegistry CODEC_REGISTRY = MongoClientSettings.getDefaultCodecRegistry();

    @Mock
    private MongoTemplate mongoTemplate;

    @InjectMocks
    private AuditLogRepositoryImpl repository;

    @Test
    @DisplayName("自证：默认 codec 下 Integer→int32、Long→int64（否则本文的类型断言恒真）")
    void defaultCodecDistinguishesInt32AndInt64() {
        assertThat(bsonTypeOf(1)).isEqualTo(BsonType.INT32);
        assertThat(bsonTypeOf(1L)).isEqualTo(BsonType.INT64);
    }

    @Test
    @DisplayName("idCandidates：int32 值域内给 Long+Integer 双候选；越界只留 int64")
    void idCandidatesCoversBothBsonEncodings() {
        assertThat(AuditLogRepositoryImpl.idCandidates(790001L)).containsExactly(790001L, 790001);
        assertThat(AuditLogRepositoryImpl.idCandidates(-1L)).containsExactly(-1L, -1);
        // 越界：python 对 ≥ 2^31 的整数同样编码 int64，无需 int32 候选（且 intValue() 会溢出）
        assertThat(AuditLogRepositoryImpl.idCandidates((long) Integer.MAX_VALUE + 1))
                .containsExactly((long) Integer.MAX_VALUE + 1);
    }

    @Test
    @DisplayName("target_id 条件：snake_case 键 + int32/int64 双类型候选 + create_time 倒序")
    void targetQueryMatchesInt32AndInt64() {
        AuditLog log = new AuditLog();
        when(mongoTemplate.find(any(Query.class), eq(AuditLog.class))).thenReturn(List.of(log));

        List<AuditLog> result = repository.listByTargetTypeAndTargetId("member", 790001L);

        assertThat(result).containsExactly(log);
        ArgumentCaptor<Query> captor = ArgumentCaptor.forClass(Query.class);
        verify(mongoTemplate).find(captor.capture(), eq(AuditLog.class));
        Query query = captor.getValue();

        Document filter = query.getQueryObject();
        assertThat(filter.get("target_type")).isEqualTo("member");
        // 值 + 类型双断言：Long（int64 口径）在前，Integer（python 历史 int32）在后
        assertThat(((Document) filter.get("target_id")).getList("$in", Object.class))
                .containsExactly(790001L, 790001);
        assertThat(bsonTypesOfCandidates(filter, "target_id"))
                .containsExactlyInAnyOrder(BsonType.INT64, BsonType.INT32);
        assertThat(query.getSortObject()).isEqualTo(new Document("create_time", -1));
    }

    @Test
    @DisplayName("operator_id 条件：同样 int32/int64 双类型（python 历史 operator_id 是 int32）")
    void operatorQueryMatchesInt32AndInt64() {
        when(mongoTemplate.find(any(Query.class), eq(AuditLog.class))).thenReturn(List.of());

        repository.listByOperatorId(2L);

        ArgumentCaptor<Query> captor = ArgumentCaptor.forClass(Query.class);
        verify(mongoTemplate).find(captor.capture(), eq(AuditLog.class));
        Document filter = captor.getValue().getQueryObject();

        assertThat(((Document) filter.get("operator_id")).getList("$in", Object.class))
                .containsExactly(2L, 2);
        assertThat(bsonTypesOfCandidates(filter, "operator_id"))
                .containsExactlyInAnyOrder(BsonType.INT64, BsonType.INT32);
        assertThat(captor.getValue().getSortObject()).isEqualTo(new Document("create_time", -1));
    }

    /** 单个值的 BSON 类型（真编码：Document → BsonDocument）。 */
    private static BsonType bsonTypeOf(Object value) {
        return new Document("v", value)
                .toBsonDocument(BsonDocument.class, CODEC_REGISTRY)
                .get("v")
                .getBsonType();
    }

    /** 过滤条件中 `$in` 候选值的 BSON 类型（用于证明两个候选分别是 int64 / int32）。 */
    private static List<BsonType> bsonTypesOfCandidates(Document filter, String field) {
        BsonDocument encoded = filter.toBsonDocument(BsonDocument.class, CODEC_REGISTRY);
        List<BsonType> types = new ArrayList<>();
        for (BsonValue value : encoded.getDocument(field).getArray("$in")) {
            types.add(value.getBsonType());
        }
        return types;
    }
}
