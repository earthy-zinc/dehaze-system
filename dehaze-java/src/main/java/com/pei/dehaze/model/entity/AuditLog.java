package com.pei.dehaze.model.entity;

import lombok.Data;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;

@Data
@Document(collection = "audit_log")
public class AuditLog implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    @Id
    private String id;

    private Long operatorId;

    private String targetType;

    private Object targetId;

    private String action;

    private String module;

    private Object beforeValue;

    private Object afterValue;

    private String ip;

    private String userAgent;

    @Indexed
    private LocalDateTime createTime;
}
