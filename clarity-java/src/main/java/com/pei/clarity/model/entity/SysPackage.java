package com.pei.clarity.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.clarity.common.base.BaseEntity;
import java.io.Serial;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_package")
public class SysPackage extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String name;

    private String packageType;

    private String levelCode;

    private String period;

    private Integer periodDays;

    private Long creditAmount;

    private Long originalPrice;

    private Long salePrice;

    private String description;

    private String benefitOverrides;

    private Long salesCount;

    private Integer sort;

    private Integer status;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}
