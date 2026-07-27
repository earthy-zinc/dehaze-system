package com.pei.dehaze.job;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysEvalLog;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.security.util.SystemSecurityContext;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;

/**
 * 预测/评估日志僵尸任务恢复
 *
 * <p>服务重启或异步线程异常退出可能残留 status=处理中 的记录，
 * 每 60 秒扫描超时（10 分钟未更新）记录并标记为 失败。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class PredEvalLogCleanupJob {

    private static final String STUCK_ERROR_MSG = "任务执行超时，服务可能已重启";

    private final SysPredLogMapper predLogMapper;
    private final SysEvalLogMapper evalLogMapper;

    @Scheduled(cron = "0 * * * * ?")
    public void cleanupStuckTasks() {
        SystemSecurityContext.setSystemContext();
        try {
            LocalDateTime threshold = LocalDateTime.now().minusMinutes(10);

            var stuckPred = new LambdaQueryWrapper<SysPredLog>()
                    .eq(SysPredLog::getStatus, LogStatusEnum.PROCESSING)
                    .lt(SysPredLog::getUpdateTime, threshold);
            int predCount = markStuckPredAsFailed(stuckPred);

            var stuckEval = new LambdaQueryWrapper<SysEvalLog>()
                    .eq(SysEvalLog::getStatus, LogStatusEnum.PROCESSING)
                    .lt(SysEvalLog::getUpdateTime, threshold);
            int evalCount = markStuckEvalAsFailed(stuckEval);

            if (predCount > 0 || evalCount > 0) {
                log.warn("清理僵尸预测/评估任务: pred={}, eval={}", predCount, evalCount);
            }
        } finally {
            SystemSecurityContext.clearContext();
        }
    }

    private int markStuckPredAsFailed(LambdaQueryWrapper<SysPredLog> wrapper) {
        var stuck = predLogMapper.selectList(wrapper);
        for (SysPredLog log : stuck) {
            SysPredLog update = new SysPredLog();
            update.setId(log.getId());
            update.setStatus(LogStatusEnum.FAILED);
            update.setErrorMessage(STUCK_ERROR_MSG);
            predLogMapper.updateById(update);
        }
        return stuck.size();
    }

    private int markStuckEvalAsFailed(LambdaQueryWrapper<SysEvalLog> wrapper) {
        var stuck = evalLogMapper.selectList(wrapper);
        for (SysEvalLog log : stuck) {
            SysEvalLog update = new SysEvalLog();
            update.setId(log.getId());
            update.setStatus(LogStatusEnum.FAILED);
            update.setErrorMessage(STUCK_ERROR_MSG);
            evalLogMapper.updateById(update);
        }
        return stuck.size();
    }
}
