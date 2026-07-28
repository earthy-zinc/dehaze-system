package com.pei.dehaze.repository;

import com.pei.dehaze.model.entity.AuditLog;
import org.springframework.data.mongodb.repository.MongoRepository;

import java.util.List;

public interface AuditLogRepository extends MongoRepository<AuditLog, String> {

    List<AuditLog> findByOperatorIdOrderByCreateTimeDesc(Long operatorId);

    List<AuditLog> findByTargetTypeAndTargetIdOrderByCreateTimeDesc(String targetType, Object targetId);
}
