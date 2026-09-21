package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

/** AI 模型成本单价版本（表 sys_ai_model_cost，供应商采购价，与用户售价 sys_ai_model_price 结构对称） */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysAiModelCost extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String modelId;

    private Long providerId;

    private Integer priceVersion;

    /** CNY/USD */
    private String currency;

    private LocalDateTime effectiveFrom;

    private LocalDateTime effectiveTo;

    private Integer status;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;
}
