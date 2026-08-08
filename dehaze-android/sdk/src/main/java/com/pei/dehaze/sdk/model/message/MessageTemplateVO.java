package com.pei.dehaze.sdk.model.message;

import lombok.Data;

import java.util.Map;

@Data
public class MessageTemplateVO {
    private Long id;
    private String code;
    private String name;
    private String type;
    private String titleTemplate;
    private String contentTemplate;
    private Integer priority;
    private Map<String, Boolean> channels;
    private Object[] variables;
    private Integer status;
    private String createTime;
    private String updateTime;
}
