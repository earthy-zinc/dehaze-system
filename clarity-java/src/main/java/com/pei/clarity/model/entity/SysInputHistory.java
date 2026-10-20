package com.pei.clarity.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.clarity.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 图像输入历史记录实体
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_input_history")
public class SysInputHistory extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    /** 用户ID */
    private Long userId;

    /** 原始图片URL */
    private String originalImageUrl;

    /** 原始缩略图URL */
    private String originalThumbnailUrl;

    /** 处理结果图片URL */
    private String resultImageUrl;

    /** 结果缩略图URL */
    private String resultThumbnailUrl;

    /** 使用的图像模型标识（统一模型规范 slug） */
    private String modelId;

    /** 模型名称（冗余字段） */
    private String modelName;

    /** 模型参数（JSON） */
    private String modelParams;

    /** 处理耗时（毫秒） */
    private Integer processingTime;

    /** 处理状态（1=成功，2=失败，3=处理中），创建时确定，不随处理进度更新 */
    private Integer status;

    /** 图片来源（upload/camera/sample） */
    private String inputSource;
}
