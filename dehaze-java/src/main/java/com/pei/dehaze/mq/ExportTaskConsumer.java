package com.pei.dehaze.mq;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.mapper.SysTaskMapper;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.service.TaskExecutor;
import com.rabbitmq.client.Channel;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

/**
 * 导出任务消费者
 * <p>
 * 消费 task.export 队列消息，调用 {@link TaskExecutor#executeExportTask} 执行任务。
 * 通过策略模式统一处理所有导出类型（dataset_export / user_export / role_export 等）。
 *
 * @author earthyzinc
 * @since 2024/4/18
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "rabbitmq", name = "enabled", havingValue = "true")
public class ExportTaskConsumer extends RabbitMQConsumer {

    private static final String QUEUE_EXPORT = "task.export";

    private final TaskExecutor taskExecutor;
    private final SysTaskMapper sysTaskMapper;

    /**
     * 消费导出任务队列
     * <p>
     * 消息体为统一 JSON 契约：{"db_task_id":123, "task_id":"uuid", "task_type":"dataset_export"}
     */
    @RabbitListener(queues = QUEUE_EXPORT)
    public void onExportMessage(Message message, Channel channel) {
        processMessage(message, channel, QUEUE_EXPORT, this::handleExport);
    }

    private void handleExport(String body, String traceId) throws Exception {
        JSONObject msg = JSONUtil.parseObj(body);
        Long dbTaskId = msg.getLong("db_task_id");

        // 幂等检查：任务非终态才执行
        SysTask sysTask = sysTaskMapper.selectById(dbTaskId);
        if (sysTask == null) {
            log.warn("任务不存在，跳过: dbTaskId={}", dbTaskId);
            return;
        }
        if (TaskConstants.TERMINAL_STATUSES.contains(sysTask.getStatus())) {
            log.debug("任务已为终态，跳过重复消费: dbTaskId={}, taskId={}, status={}",
                    dbTaskId, sysTask.getTaskId(), sysTask.getStatus());
            return;
        }

        log.debug("收到导出任务: dbTaskId={}, taskId={}, type={}, traceId={}",
                dbTaskId, sysTask.getTaskId(), sysTask.getTaskType(), traceId);
        // form 传 null，由 executeExportTask 从 sysTask.getParams() 重建
        taskExecutor.executeExportTask(dbTaskId, null);
        log.debug("导出任务处理完成: dbTaskId={}, traceId={}", dbTaskId, traceId);
    }
}
