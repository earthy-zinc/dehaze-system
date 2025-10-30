package com.pei.dehaze.module.iot.dal.dataobject.device;

import com.baomidou.mybatisplus.annotation.KeySequence;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.framework.mybatis.core.dataobject.BaseDO;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * IoT 设备分组 DO
 *
 * @author earthyzinc
 */
@TableName("iot_device_group")
@KeySequence("iot_device_group_seq") // 用于 Oracle、PostgreSQL、Kingbase、DB2、H2 数据库的主键自增。如果是 MySQL 等数据库，可不写。
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class IotDeviceGroupDO extends BaseDO {

    /**
     * 分组 ID
     */
    @TableId
    private Long id;
    /**
     * 分组名字
     */
    private String name;
    /**
     * 分组状态
     * <p>
     * 枚举 {@link com.pei.dehaze.framework.common.enums.CommonStatusEnum}
     */
    private Integer status;
    /**
     * 分组描述
     */
    private String description;

}
