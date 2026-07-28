package com.pei.dehaze.job;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.mapper.SysMessageMapper;
import com.pei.dehaze.model.entity.SysMessage;
import com.pei.dehaze.security.util.SystemSecurityContext;
import com.xxl.job.core.handler.annotation.XxlJob;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 过期消息定时清理任务
 * 每天凌晨4点清理过期消息（expires_at < NOW()），分批处理每批500条。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class MessageCleanupJob {

    private static final int BATCH_SIZE = 500;

    private final SysMessageMapper messageMapper;

    @XxlJob("cleanupExpiredMessages")
    public void cleanupExpiredMessages() {
        SystemSecurityContext.setSystemContext();
        try {
            log.info("开始清理过期消息...");
            int totalDeleted = 0;
            while (true) {
                List<Long> ids = messageMapper.selectList(new LambdaQueryWrapper<SysMessage>()
                                .select(SysMessage::getId)
                                .lt(SysMessage::getExpiresAt, LocalDateTime.now())
                                .last("LIMIT " + BATCH_SIZE))
                        .stream()
                        .map(SysMessage::getId)
                        .toList();
                if (ids.isEmpty()) {
                    break;
                }
                messageMapper.physicalDeleteByIds(ids);
                totalDeleted += ids.size();
            }
            if (totalDeleted > 0) {
                log.info("清理过期消息完成: 共清理{}条", totalDeleted);
            }
        } finally {
            SystemSecurityContext.clearContext();
        }
    }
}
