package com.pei.dehaze.mq;

import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.security.util.SystemSecurityContext;
import com.pei.dehaze.service.TaskExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.MDC;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import com.pei.dehaze.filter.TraceIdFilter;

import java.util.Set;

/**
 * 导出任务消费者
 * <p>
 * 消费 task.export 队列消息，调用 {@link TaskExecutor#executeExportTask} 执行任务。
 * 通过策略模式统一处理所有导出类型（dataset_export / item_download / batch_download / custom_export）。
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "rabbitmq", name = "enabled", havingValue = "true")
public class ExportTaskConsumer extends RabbitMQConsumer {

    private static final Set<String> TERMINAL_STATUSES = Set.of(
            TaskConstants.STATUS_COMPLETED,
            TaskConstants.STATUS_FAILED,
            TaskConstants.STATUS_CANCELLED
    );

    private final TaskExecutor taskExecutor;
    private final SysTaskMapper sysTaskMapper;

    /**
     * 消费导出任务队列
     * <p>
     * 消息体为 taskId（数据库主键），Consumer 从 DB 重建 form 后执行。
     */
    @RabbitListener(queues = "task.export")
    public void onExportMessage(Message message) {
        SystemSecurityContext.setSystemContext();
        String traceId = extractTraceId(message);
        // 传播 TraceId 到 MDC（异步上下文）
        if (traceId != null) {
            MDC.put(TraceIdFilter.MDC_TRACE_ID, traceId);
        }
        try {
            String body = extractBody(message);
            Long taskId = Long.parseLong(body.trim());

            // 幂等检查：任务非终态才执行
            SysTask sysTask = sysTaskMapper.selectById(taskId);
            if (sysTask == null) {
                log.warn("任务不存在，跳过: taskId={}", taskId);
                return;
            }
            if (TERMINAL_STATUSES.contains(sysTask.getStatus())) {
                log.info("任务已为终态，跳过重复消费: taskId={}, status={}", taskId, sysTask.getStatus());
                return;
            }

            log.info("收到导出任务: taskId={}, traceId={}", taskId, traceId);
            // form 传 null，由 executeExportTask 从 sysTask.getParams() 重建
            taskExecutor.executeExportTask(taskId, null);
            log.debug("导出任务处理完成: taskId={}, traceId={}", taskId, traceId);
        } catch (Exception e) {
            handleError("task.export", traceId, e);
        } finally {
            MDC.remove(TraceIdFilter.MDC_TRACE_ID);
            SystemSecurityContext.clearContext();
        }
    }
}
