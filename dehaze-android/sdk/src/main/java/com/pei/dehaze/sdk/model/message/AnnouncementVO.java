package com.pei.dehaze.sdk.model.message;

import lombok.Data;

import java.util.Map;

@Data
public class AnnouncementVO {
    private Long id;
    private String title;
    private String content;
    private String type;
    private String typeLabel;
    private Integer importance;
    private String importanceLabel;
    private String targetScope;
    private String targetScopeLabel;
    private Map<String, Object> targetParams;
    private Integer status;
    private String statusLabel;
    private String sendTime;
    private String expireTime;
    private Integer sentCount;
    private String createTime;
    private String updateTime;
    private Long createBy;
}
