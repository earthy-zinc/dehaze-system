package com.pei.dehaze.sdk.model.message;

import lombok.Data;

import java.util.Map;

@Data
public class AnnouncementForm {
    private String title;
    private String content;
    private String type;
    private Integer importance;
    private String targetScope;
    private Map<String, Object> targetParams;
    private String sendTime;
    private String expireTime;
}
