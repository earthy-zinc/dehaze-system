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
 * AI 长期记忆
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_ai_memory", autoResultMap = true)
public class SysAiMemory extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private String memoryType;

    private String content;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private Map<String, Object> metadata;

    private Integer importance;

    private Integer accessCount;

    private LocalDateTime lastAccessedAt;

    private String source;

    private Integer status;

    private Integer archived;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    private LocalDateTime deleteTime;

    @Serial
    private static final long serialVersionUID = 1L;
}
