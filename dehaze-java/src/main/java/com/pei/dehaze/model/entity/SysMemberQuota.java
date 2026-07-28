package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;

@Data
@TableName("sys_member_quota")
public class SysMemberQuota implements Serializable {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private Integer quotaMonth;

    private String levelCode;

    private Integer dehazeQuota;

    private Integer dehazeUsed;

    private Integer evaluateQuota;

    private Integer evaluateUsed;

    private LocalDateTime resetTime;

    private LocalDateTime createTime;

    @Serial
    private static final long serialVersionUID = 1L;
}
