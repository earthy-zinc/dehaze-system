package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import com.pei.dehaze.common.enums.LogStatusEnum;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
public class SysPredLog extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long algorithmId;

    private Long originFileId;

    private String originMd5;

    private String originUrl;

    private Long predFileId;

    private String predMd5;

    private String predUrl;

    private Integer time;

    private LogStatusEnum status;

    private String errorMessage;
}
