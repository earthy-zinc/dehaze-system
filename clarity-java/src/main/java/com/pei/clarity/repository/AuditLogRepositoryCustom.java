package com.pei.clarity.repository;

import com.pei.clarity.model.entity.AuditLog;
import java.util.List;

/**
 * {@code audit_log} 集合的自定义查询片段（跨端互认口径）。
 *
 * <p>该集合由 java / go / clarity-python 三端共写共读，而同一整数的 BSON 编码随端而异：
 * java（{@code Long}）与 go（{@code int64}）写 int64，python(pymongo) 对绝对值 &lt; 2^31
 * 的整数写 <b>int32</b>。BSON 数值类型不相等，单类型派生查询读不到 python 写入的记录，
 * <b>且不抛错</b>（静默空列表）。故数值归属字段一律走 int32/int64 双类型匹配
 * （等价 go {@code audit_log_repository.ListByTarget} 的 {@code bson.A{int32(id), id}}）。
 */
public interface AuditLogRepositoryCustom {

    /** 按目标对象查询审计日志（create_time 倒序），{@code target_id} 双类型匹配。 */
    List<AuditLog> listByTargetTypeAndTargetId(String targetType, Long targetId);

    /** 按操作人查询审计日志（create_time 倒序），{@code operator_id} 双类型匹配。 */
    List<AuditLog> listByOperatorId(Long operatorId);
}
