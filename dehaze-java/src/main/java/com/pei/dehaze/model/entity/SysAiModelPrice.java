package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import java.time.LocalDateTime;
import lombok.Data;
import lombok.EqualsAndHashCode;

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

    /** 计费计量单位(million_tokens:百万token;image:张;second:秒;ten_thousand_chars:万字符;frame:帧) */
    private String billingUnit;

    private LocalDateTime effectiveFrom;

    private LocalDateTime effectiveTo;

    private Integer status;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;
}
