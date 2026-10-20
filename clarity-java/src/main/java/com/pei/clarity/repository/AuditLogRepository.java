package com.pei.clarity.repository;

import com.pei.clarity.model.entity.AuditLog;
import org.springframework.data.mongodb.repository.MongoRepository;

/**
 * 业务操作审计日志（MongoDB 集合 {@code audit_log}）。
 *
 * <p>集合由 java / go / clarity-python 共写共读。数值归属字段（{@code operator_id} /
 * {@code target_id}）不能用按属性类型的派生查询——单类型条件会漏读对端与历史记录
 * 且不报错，故改由 {@link AuditLogRepositoryCustom} 提供 int32/int64 双类型匹配。
 */
public interface AuditLogRepository extends MongoRepository<AuditLog, String>, AuditLogRepositoryCustom {}
