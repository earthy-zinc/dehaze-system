package com.pei.dehaze.controller;

import com.pei.dehaze.model.dto.ChatMessage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.messaging.handler.annotation.DestinationVariable;
import org.springframework.messaging.handler.annotation.MessageMapping;
import org.springframework.messaging.handler.annotation.SendTo;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.security.Principal;

/**
 * WebSocket 测试控制器
 *
 * @author earthyzinc
 * @since 2.3.0
 */
@RestController
@RequestMapping("/websocket")
@RequiredArgsConstructor
@Slf4j
public class WebsocketController {

    private final SimpMessagingTemplate messagingTemplate;


    /**
     * 广播发送消息
     *
     * @param message 消息内容
     */
    @MessageMapping("/sendToAll")
    @SendTo("/topic/notice")
    public String sendToAll(String message) {
        return "服务端通知: " + message;
    }

    /**
     * 点对点发送消息
     * <p>
     * 模拟用户A 给 用户B 发送消息场景
     *
     * @param principal 当前用户（Principal name = userId）
     * @param userId    接收消息的用户 ID
     * @param message   消息内容
     */
    @MessageMapping("/sendToUser/{userId}")
    public void sendToUser(Principal principal, @DestinationVariable String userId, String message) {

        String senderId = principal.getName(); // 发送人 userId

        log.debug("发送人 userId:{}; 接收人 userId:{}", senderId, userId);
        // 发送消息给指定用户，拼接后路径 /user/{receiver}/queue/greeting
        messagingTemplate.convertAndSendToUser(userId, "/queue/greeting", new ChatMessage(senderId, message));
    }

}
