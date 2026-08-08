package com.pei.dehaze.sdk.model.message;

import lombok.Data;

import java.util.Map;

@Data
public class MessageVO {
    private Long id;
    private String type;
    private String typeLabel;
    private String title;
    private String summary;
    private String content;
    private Integer priority;
    private Integer readStatus;
    private Integer senderType;
    private String readTime;
    private String jumpUrl;
    private Map<String, Object> extra;
    private String createTime;
}
