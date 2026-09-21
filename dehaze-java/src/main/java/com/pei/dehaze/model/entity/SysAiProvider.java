package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * AI 模型供应商（表 sys_ai_provider）。
 *
 * <p>provider_code 为白名单业务键：软删后不可复用，查重必须查全表（含软删行）。
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysAiProvider extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String providerCode;

    private String displayName;

    private String apiBaseUrl;

    private String protocolType;

    private String authType;

    private String defaultHeaders;

    private Integer sortOrder;

    private Integer healthCheckEnabled;

    private String userIdentityForward;

    private String remark;

    private Integer status;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;
}
