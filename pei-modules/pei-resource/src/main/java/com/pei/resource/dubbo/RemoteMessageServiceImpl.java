package com.pei.resource.dubbo;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboService;
import com.pei.common.sse.dto.SseMessageDto;
import com.pei.common.sse.utils.SseMessageUtils;
import com.pei.resource.api.RemoteMessageService;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 消息服务
 *
 * @author Lion Li
 */
@Slf4j
@RequiredArgsConstructor
@Service
@DubboService
public class RemoteMessageServiceImpl implements RemoteMessageService {

    /**
     * 发送消息
     *
     * @param sessionKey session主键 一般为用户id
     * @param message    消息文本
     */
    @Override
    public void publishMessage(List<Long> sessionKey, String message) {
        SseMessageDto dto = new SseMessageDto();
        dto.setMessage(message);
        dto.setUserIds(sessionKey);
        SseMessageUtils.publishMessage(dto);
    }

    /**
     * 发布订阅的消息(群发)
     *
     * @param message 消息内容
     */
    @Override
    public void publishAll(String message) {
        SseMessageUtils.publishAll(message);
    }

}
