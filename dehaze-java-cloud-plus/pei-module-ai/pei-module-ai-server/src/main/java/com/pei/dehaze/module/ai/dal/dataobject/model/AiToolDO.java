package com.pei.dehaze.module.ai.dal.dataobject.model;

import com.pei.dehaze.framework.mybatis.core.dataobject.BaseDO;
import com.pei.dehaze.module.ai.service.model.tool.DirectoryListToolFunction;
import com.pei.dehaze.module.ai.service.model.tool.WeatherQueryToolFunction;
import com.baomidou.mybatisplus.annotation.KeySequence;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.*;

/**
 * AI 工具 DO
 *
 * @author earthyzinc
 */
@TableName("ai_tool")
@KeySequence("ai_tool_seq") // 用于 Oracle、PostgreSQL、Kingbase、DB2、H2 数据库的主键自增。如果是 MySQL 等数据库，可不写。
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AiToolDO extends BaseDO {

    /**
     * 工具编号
     */
    @TableId
    private Long id;
    /**
     * 工具名称
     *
     * 对应 Bean 的名字，例如说：
     * 1. {@link DirectoryListToolFunction} 的 Bean 名字是 directory_list
     * 2. {@link WeatherQueryToolFunction} 的 Bean 名字是 weather_query
     */
    private String name;
    /**
     * 工具描述
     */
    private String description;
    /**
     * 状态
     *
     * 枚举 {@link com.pei.dehaze.framework.common.enums.CommonStatusEnum}
     */
    private Integer status;

}
