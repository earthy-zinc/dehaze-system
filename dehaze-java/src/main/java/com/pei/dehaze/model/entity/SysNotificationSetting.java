package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;

import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;
import java.time.LocalTime;

@Data
@TableName("sys_notification_setting")
public class SysNotificationSetting implements Serializable {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private Integer pushEnabled;

    private Integer dndEnabled;

    private LocalTime dndStart;

    private LocalTime dndEnd;

    private String preferences;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;

    @Serial
    private static final long serialVersionUID = 1L;
}
