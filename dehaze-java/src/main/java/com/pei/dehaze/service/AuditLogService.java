package com.pei.dehaze.service;

public interface AuditLogService {

    void recordAudit(Long operatorId, String targetType, Object targetId, String action, String module, Object beforeValue, Object afterValue, String ip, String userAgent);
}
