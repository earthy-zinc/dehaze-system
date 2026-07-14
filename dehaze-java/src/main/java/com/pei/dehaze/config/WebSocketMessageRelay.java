package com.pei.dehaze.config;

import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.constant.TaskConstants;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.connection.Message;
import org.springframework.data.redis.connection.MessageListener;
import org.springframework.data.redis.listener.ChannelTopic;
import org.springframework.data.redis.listener.RedisMessageListenerContainer;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

/**
 * WebSocket 消息中继
 * <p>
 * 通过 Redis Pub/Sub 实现跨实例 WebSocket 消息投递，对齐 Python 端方案。
 * 任务执行线程发布消息到 Redis 频道，每个实例订阅该频道并将消息投递给本地 STOMP 连接。
 *
 * @author earthy-zinc
 * @since 2026-07-14
 */
@Slf4j
@Component
public class WebSocketMessageRelay implements MessageListener {

    @Resource
    private StringRedisTemplate stringRedisTemplate;

    @Resource
    private SimpMessagingTemplate messagingTemplate;

    @Resource
    private RedisMessageListenerContainer redisMessageListenerContainer;

    @PostConstruct
    public void init() {
        redisMessageListenerContainer.addMessageListener(this,
                new ChannelTopic(TaskConstants.WS_CHANNEL));
        log.info("WebSocket 消息中继已订阅频道: {}", TaskConstants.WS_CHANNEL);
    }

    /**
     * 发布任务消息到指定用户（跨实例）
     *
     * @param userId  目标用户 ID
     * @param message WebSocket 消息体
     */
    public void publishToUser(Long userId, Map<String, Object> message) {
        try {
            Map<String, Object> envelope = new HashMap<>();
            envelope.put("target_user_id", userId);
            envelope.put("message", message);
            stringRedisTemplate.convertAndSend(TaskConstants.WS_CHANNEL, JSONUtil.toJsonStr(envelope));
        } catch (Exception e) {
            log.debug("Redis Pub/Sub 发布失败（不影响任务执行）: {}", e.getMessage());
        }
    }

    /**
     * 接收 Redis Pub/Sub 消息并投递给本地 STOMP 连接
     */
    @Override
    public void onMessage(Message redisMessage, byte[] pattern) {
        try {
            String json = new String(redisMessage.getBody());
            Map<String, Object> envelope = JSONUtil.toBean(json, Map.class);
            if (envelope == null) {
                return;
            }

            Object userIdObj = envelope.get("target_user_id");
            if (userIdObj == null) {
                return;
            }
            String userId = String.valueOf(userIdObj);

            @SuppressWarnings("unchecked")
            Map<String, Object> message = (Map<String, Object>) envelope.get("message");
            if (message == null) {
                return;
            }

            messagingTemplate.convertAndSendToUser(userId, "/queue/task", message);
        } catch (Exception e) {
            log.debug("WebSocket 消息本地投递失败: {}", e.getMessage());
        }
    }
}
