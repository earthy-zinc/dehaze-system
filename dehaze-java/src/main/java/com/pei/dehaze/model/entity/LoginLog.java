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
 * 登录审计日志（MongoDB 集合 {@code login_log}）。
 *
 * <p>集合由 java 与 dehaze-python 共写共读，python 写入的文档键为 snake_case，
 * 故字段必须显式 {@link Field} 映射，否则同集合内两端数据互不可见。
 */
@Data
@Document(collection = "login_log")
public class LoginLog implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    @Id
    private String id;

    @Field("user_id")
    private Long userId;

    private String username;

    private String ip;

    private String location;

    private String browser;

    private String os;

    @Field("device_type")
    private String deviceType;

    private Integer status;

    private String message;

    @Indexed
    @Field("create_time")
    private LocalDateTime createTime;
}
