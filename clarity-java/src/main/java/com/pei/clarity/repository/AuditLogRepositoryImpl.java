package com.pei.clarity.repository;

import com.pei.clarity.common.util.MongoBsonUtils;
import com.pei.clarity.model.entity.AuditLog;
import java.util.List;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Sort;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;

/**
 * {@link AuditLogRepositoryCustom} 的实现（Spring Data 自定义片段）。
 *
 * <p>数值归属字段（{@code operator_id} / {@code target_id}）用 int32/int64 <b>双类型</b>
 * {@code $in}（候选由 {@link MongoBsonUtils#int32Int64Candidates} 统一给出）：java/go 写入并查询的是
 * int64，python(pymongo) 写的是 int32（绝对值 &lt; 2^31 时），单类型条件会静默漏掉对端与历史记录。
 * 详见 {@link AuditLogRepositoryCustom}。
 */
@RequiredArgsConstructor
public class AuditLogRepositoryImpl implements AuditLogRepositoryCustom {

    private final MongoTemplate mongoTemplate;

    @Override
    public List<AuditLog> listByTargetTypeAndTargetId(String targetType, Long targetId) {
        return findAll(targetQuery(targetType, targetId));
    }

    @Override
    public List<AuditLog> listByOperatorId(Long operatorId) {
        return findAll(operatorQuery(operatorId));
    }

    private List<AuditLog> findAll(Query query) {
        return mongoTemplate.find(query, AuditLog.class);
    }

    /** 目标对象查询（create_time 倒序），条件键名取实体 {@code @Field} 映射后的 snake_case。 */
    static Query targetQuery(String targetType, Long targetId) {
        Criteria criteria =
                Criteria.where("target_type").is(targetType).and("target_id").in(idCandidates(targetId));
        return new Query(criteria).with(Sort.by(Sort.Direction.DESC, "create_time"));
    }

    /** 操作人查询（create_time 倒序）。 */
    static Query operatorQuery(Long operatorId) {
        Criteria criteria = Criteria.where("operator_id").in(idCandidates(operatorId));
        return new Query(criteria).with(Sort.by(Sort.Direction.DESC, "create_time"));
    }

    /**
     * 数值归属字段的 BSON 双类型候选（{@link MongoBsonUtils#int32Int64Candidates}）。
     *
     * <p>{@code Long} 由驱动落 int64（java/go 口径）、{@code Integer} 落 int32
     * （python 历史记录与 pymongo 的小整数口径）；超出 int32 值域时只保留 int64
     * ——python 对 ≥ 2^31 的整数同样编码为 int64。
     */
    static List<Object> idCandidates(Long id) {
        return MongoBsonUtils.int32Int64Candidates(id);
    }
}
