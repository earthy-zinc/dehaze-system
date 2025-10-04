package com.pei.dehaze.module.iot.dal.dataobject.rule;

import com.pei.dehaze.framework.common.enums.CommonStatusEnum;
import com.pei.dehaze.framework.mybatis.core.dataobject.BaseDO;
import com.pei.dehaze.module.iot.controller.admin.rule.vo.databridge.config.IotDataBridgeAbstractConfig;
import com.pei.dehaze.module.iot.enums.rule.IotDataBridgeDirectionEnum;
import com.pei.dehaze.module.iot.enums.rule.IotDataBridgeTypeEnum;
import com.baomidou.mybatisplus.annotation.KeySequence;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import lombok.*;

/**
 * IoT 数据桥梁 DO
 *
 * @author earthyzinc
 */
@TableName(value = "iot_data_bridge", autoResultMap = true)
@KeySequence("iot_data_bridge_seq") // 用于 Oracle、PostgreSQL、Kingbase、DB2、H2 数据库的主键自增。如果是 MySQL 等数据库，可不写。
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class IotDataBridgeDO extends BaseDO {

    /**
     * 桥梁编号
     */
    @TableId
    private Long id;
    /**
     * 桥梁名称
     */
    private String name;
    /**
     * 桥梁描述
     */
    private String description;
    /**
     * 桥梁状态
     *
     * 枚举 {@link CommonStatusEnum}
     */
    private Integer status;
    /**
     * 桥梁方向
     *
     * 枚举 {@link IotDataBridgeDirectionEnum}
     */
    private Integer direction;

    /**
     * 桥梁类型
     *
     * 枚举 {@link IotDataBridgeTypeEnum}
     */
    private Integer type;

    /**
     * 桥梁配置
     */
    @TableField(typeHandler = JacksonTypeHandler.class)
    private IotDataBridgeAbstractConfig config;

}
