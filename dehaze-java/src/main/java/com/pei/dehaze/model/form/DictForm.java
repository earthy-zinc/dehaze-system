package com.pei.dehaze.model.form;


import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Schema(description = "字典表单对象")
@Data
public class DictForm {

    @Schema(description="字典ID")
    private Long id;

    @Schema(description="类型编码")
    @NotBlank(message = "字典类型编码不能为空")
    private String typeCode;

    @Schema(description="字典名称")
    @NotBlank(message = "字典名称不能为空")
    private String name;

    @Schema(description="字典值")
    @NotBlank(message = "字典值不能为空")
    private String value;

    @Schema(description="状态(1:启用;0:禁用)")
    private Integer status;

    @Schema(description="排序")
    private Integer sort;

    @Schema(description="是否默认(1:是;0:否)")
    private Integer defaulted;

    @Schema(description = "字典备注")
    private String remark;

}
