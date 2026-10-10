package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import java.io.Serial;
import java.time.LocalDateTime;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_rating")
public class SysRating extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private Long predLogId;

    private String modelId;

    private Integer rating;

    private String comment;

    private String tags;

    private String imageUrls;

    private Integer isAnonymous;

    private Integer isHidden;

    private String adminReply;

    private LocalDateTime replyTime;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
