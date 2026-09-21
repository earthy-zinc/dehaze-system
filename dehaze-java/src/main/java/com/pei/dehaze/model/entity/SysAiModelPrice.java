package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

/** AI 模型用户售价版本（表 sys_ai_model_price） */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysAiModelPrice extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String modelId;

    private Long providerId;

    private Integer priceVersion;

    private String unit;

    private LocalDateTime effectiveFrom;

    private LocalDateTime effectiveTo;

    private Integer status;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;
}
