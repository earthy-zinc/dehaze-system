package com.pei.dehaze.sdk.model.message;

import lombok.Data;

import java.util.List;

@Data
public class MessageSendRequest {
    private String templateCode;
    private String type;
    private String title;
    private String content;
    private List<Long> recipientIds;
    private String bizModule;
    private String bizId;
    private Integer priority;
    private String jumpUrl;
    private Object variables;
    private Object extra;
}
