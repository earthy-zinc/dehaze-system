package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import java.io.Serial;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 用户 AI 偏好设置（表 sys_ai_user_preference，每用户一行）。
 *
 * <p>仅承载"面向用户"的默认对话模型；后台派生推理不读本表。
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_ai_user_preference")
public class SysAiUserPreference extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    /** 偏好默认对话模型（存 sys_ai_model.model_id；NULL 表示未设置，解析时回落平台默认模型） */
    private String defaultModel;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
