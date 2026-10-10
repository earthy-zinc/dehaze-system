package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import com.pei.dehaze.common.enums.LogStatusEnum;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 图像任务日志（表 sys_pred_log）。
 *
 * <p>model_id 为统一模型规范 slug，provider_id/actual_model 记录实际调用供应商与模型（降级/路由后可能与请求不同），
 * vendor_task_id 记录异步厂商任务号（本地同步推理为空）。
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysPredLog extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String modelId;

    private Long providerId;

    private String actualModel;

    private String vendorTaskId;

    private Long originFileId;

    private String originMd5;

    private String originUrl;

    private Long predFileId;

    private String predMd5;

    private String predUrl;

    private Long recommendedBy;

    private Integer time;

    private LogStatusEnum status;

    private String errorMessage;
}
