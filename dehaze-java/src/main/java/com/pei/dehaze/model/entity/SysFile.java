package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;

@Data
@EqualsAndHashCode(callSuper = false)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SysFile extends BaseEntity {
    @TableId(type = IdType.AUTO)
    private Long id;

    private String type;

    private String name;

    private String objectName;

    /**
     * 存储后端标识（minio / local / nginx-static），与环境无关
     */
    private String storage;

    /**
     * 文件大小（格式化显示，如 "2.44MB"）
     */
    private String size;

    /**
     * 文件大小（原始字节数）
     */
    private Long sizeBytes;

    private String md5;

    @TableLogic
    private Integer deleted;

    /**
     * 文件访问 URL（运行时动态拼接，不落库）
     */
    @TableField(exist = false)
    private String url;
}
