package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 算法版本历史表
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_algorithm_version")
public class SysAlgorithmVersion extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 关联算法ID
     */
    private Long algorithmId;

    /**
     * 版本号（语义化版本，如 1.0.0）
     */
    private String version;

    /**
     * 变更日志
     */
    private String changeLog;

    /**
     * 该版本时的状态
     */
    private Integer status;

    /**
     * 该版本时的配置JSON
     */
    private String configJson;

    /**
     * 模型文件ID
     */
    private Long modelFileId;

    /**
     * 是否当前活跃版本
     */
    private Boolean isActive;
}
