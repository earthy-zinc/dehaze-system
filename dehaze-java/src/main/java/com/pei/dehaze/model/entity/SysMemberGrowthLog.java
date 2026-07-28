package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;

@Data
@TableName("sys_member_growth_log")
public class SysMemberGrowthLog implements Serializable {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private String changeType;

    private Integer changeValue;

    private Long balance;

    private String relatedId;

    private String reason;

    private Long operatorId;

    private LocalDateTime createTime;

    @Serial
    private static final long serialVersionUID = 1L;
}
