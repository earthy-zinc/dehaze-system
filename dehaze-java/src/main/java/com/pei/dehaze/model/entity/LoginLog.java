package com.pei.dehaze.model.entity;

import lombok.Data;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;

@Data
@Document(collection = "login_log")
public class LoginLog implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    @Id
    private String id;

    private Long userId;

    private String username;

    private String ip;

    private String location;

    private String browser;

    private String os;

    private Integer status;

    private String message;

    @Indexed
    private LocalDateTime createTime;
}
