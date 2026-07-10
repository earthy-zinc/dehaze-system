package com.pei.dehaze.model.vo;


import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Schema(description ="字典分页对象")
@Data
public class DictPageVO {

    @Schema(description="字典ID")
    private Long id;

    @Schema(description="字典名称")
    private String name;

    @Schema(description="字典值")
    private String value;

    @Schema(description="字典类型编码")
    private String typeCode;

    @Schema(description="是否默认(1:是;0:否)")
    private Integer defaulted;

    @Schema(description="排序")
    private Integer sort;

    @Schema(description="状态(1:启用;0:禁用)")
    private Integer status;

    @Schema(description="备注")
    private String remark;

    @Schema(description="创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

}
