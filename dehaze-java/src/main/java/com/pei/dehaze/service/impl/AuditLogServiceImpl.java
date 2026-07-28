package com.pei.dehaze.service.impl;

import com.pei.dehaze.model.entity.AuditLog;
import com.pei.dehaze.repository.AuditLogRepository;
import com.pei.dehaze.service.AuditLogService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

@Slf4j
@Service
@RequiredArgsConstructor
public class AuditLogServiceImpl implements AuditLogService {

    private final AuditLogRepository auditLogRepository;

    @Async("datasetTaskExecutor")
    @Override
    public void recordAudit(Long operatorId, String targetType, Object targetId, String action, String module, Object beforeValue, Object afterValue, String ip, String userAgent) {
        try {
            AuditLog auditLog = new AuditLog();
            auditLog.setOperatorId(operatorId);
            auditLog.setTargetType(targetType);
            auditLog.setTargetId(targetId);
            auditLog.setAction(action);
            auditLog.setModule(module);
            auditLog.setBeforeValue(beforeValue);
            auditLog.setAfterValue(afterValue);
            auditLog.setIp(ip);
            auditLog.setUserAgent(userAgent);
            auditLog.setCreateTime(LocalDateTime.now());
            auditLogRepository.save(auditLog);
        } catch (Exception e) {
            log.warn("写入审计日志失败: operatorId={}, targetType={}, targetId={}, action={}",
                    operatorId, targetType, targetId, action, e);
        }
    }
}
