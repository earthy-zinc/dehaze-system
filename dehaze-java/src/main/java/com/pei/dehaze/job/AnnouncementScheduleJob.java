package com.pei.dehaze.job;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.model.entity.SysAnnouncement;
import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.AnnouncementService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 定时公告发送任务
 * 每分钟扫描待发送公告（status=2 且 send_time <= NOW()），逐条调用发送。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class AnnouncementScheduleJob {

    private final AnnouncementService announcementService;

    @Scheduled(cron = "0 * * * * ?")
    public void sendScheduledAnnouncements() {
        SystemSecurityContext.setSystemContext();
        try {
            List<SysAnnouncement> pending = announcementService.list(new LambdaQueryWrapper<SysAnnouncement>()
                    .eq(SysAnnouncement::getStatus, 2)
                    .le(SysAnnouncement::getSendTime, LocalDateTime.now()));
            for (SysAnnouncement announcement : pending) {
                try {
                    announcementService.send(announcement.getId());
                    log.info("定时公告发送成功: id={}", announcement.getId());
                } catch (Exception e) {
                    log.error("定时公告发送失败: id={}", announcement.getId(), e);
                }
            }
        } finally {
            SystemSecurityContext.clearContext();
        }
    }
}
