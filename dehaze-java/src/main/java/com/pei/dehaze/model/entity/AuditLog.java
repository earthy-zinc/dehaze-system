package com.pei.dehaze.model.entity;

import lombok.Data;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * 业务操作审计日志（MongoDB 集合 {@code audit_log}）。
 *
 * <p>集合由 java 与 dehaze-python 共写共读，python 写入的文档键为 snake_case，
 * 故字段必须显式 {@link Field} 映射，否则同集合内两端数据互不可见。
 */
@Data
@Document(collection = "audit_log")
public class AuditLog implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    @Id
    private String id;

    @Field("operator_id")
    private Long operatorId;

    @Field("target_type")
    private String targetType;

    @Field("target_id")
    private Object targetId;

    private String action;

    private String module;

    @Field("before_value")
    private Object beforeValue;

    @Field("after_value")
    private Object afterValue;

    private String ip;

    @Field("user_agent")
    private String userAgent;

    @Indexed
    @Field("create_time")
    private LocalDateTime createTime;
}
