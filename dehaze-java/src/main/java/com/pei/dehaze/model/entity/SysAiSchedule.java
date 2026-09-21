package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;
import java.time.LocalDateTime;
import java.util.Map;

/**
 * AI 定时任务
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_ai_schedule", autoResultMap = true)
public class SysAiSchedule extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private String name;

    private String cron;

    private String timezone;

    /** input/output 是 SQL 关键字，必须带反引号，否则 MyBatis-Plus 自动生成的列清单会被 JSQLParser 解析失败 */
    @TableField(value = "`input`", typeHandler = JacksonTypeHandler.class)
    private Map<String, Object> input;

    @TableField(value = "`output`", typeHandler = JacksonTypeHandler.class)
    private Map<String, Object> output;

    private Integer enabled;

    private Integer status;

    private Integer circuitStreak;

    private LocalDateTime nextTriggerTime;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
