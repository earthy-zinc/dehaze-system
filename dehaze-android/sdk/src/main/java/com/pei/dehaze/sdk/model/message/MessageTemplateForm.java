package com.pei.dehaze.sdk.model.message;

import lombok.Data;

import java.util.Map;

@Data
public class MessageTemplateForm {
    private String name;
    private String titleTemplate;
    private String contentTemplate;
    private Integer priority;
    private Map<String, Boolean> channels;
    private Integer status;
}
