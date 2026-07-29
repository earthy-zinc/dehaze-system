package com.pei.dehaze.config;

import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.constant.TaskConstants;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
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
@RequiredArgsConstructor
public class WebSocketMessageRelay implements MessageListener {

    public static final String DEST_TASK = "/queue/task";
    public static final String DEST_MESSAGE = "/queue/message";

    private final StringRedisTemplate stringRedisTemplate;

    private final SimpMessagingTemplate messagingTemplate;

    private final RedisMessageListenerContainer redisMessageListenerContainer;

    @PostConstruct
    public void init() {
        redisMessageListenerContainer.addMessageListener(this,
                new ChannelTopic(TaskConstants.WS_CHANNEL));
        log.info("WebSocket 消息中继已订阅频道: {}", TaskConstants.WS_CHANNEL);
    }

    /**
     * 发布消息到指定用户的指定 STOMP 队列（跨实例）
     *
     * @param userId      目标用户 ID
     * @param destination STOMP 目的地（如 /queue/task、/queue/message）
     * @param message     WebSocket 消息体
     */
    public void publishToUser(Long userId, String destination, Map<String, Object> message) {
        try {
            Map<String, Object> envelope = new HashMap<>();
            envelope.put("target_user_id", userId);
            envelope.put("destination", destination);
            envelope.put("message", message);
            stringRedisTemplate.convertAndSend(TaskConstants.WS_CHANNEL, JSONUtil.toJsonStr(envelope));
        } catch (Exception e) {
            log.warn("Redis Pub/Sub 发布失败（不影响主流程）: userId={}, dest={}", userId, destination, e);
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

            Object destObj = envelope.get("destination");
            String destination = destObj != null ? String.valueOf(destObj) : DEST_TASK;

            @SuppressWarnings("unchecked")
            Map<String, Object> message = (Map<String, Object>) envelope.get("message");
            if (message == null) {
                return;
            }

            messagingTemplate.convertAndSendToUser(userId, destination, message);
        } catch (Exception e) {
            log.warn("WebSocket 消息本地投递失败", e);
        }
    }
}
