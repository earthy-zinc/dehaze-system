package com.pei.dehaze.model.form;


import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Schema(description = "字典类型")
@Data
public class DictTypeForm {

    @Schema(description="字典类型ID")
    private Long id;

    @Schema(description="类型名称")
    @NotBlank(message = "类型名称不能为空")
    @Size(max = 64, message = "类型名称长度不能超过64")
    private String name;

    @Schema(description="类型编码")
    @NotBlank(message = "类型编码不能为空")
    @Size(max = 32, message = "类型编码长度不能超过32")
    private String code;

    @Schema(description="类型状态(1:启用;0:禁用)")
    private Integer status;

    @Schema(description = "备注")
    @Size(max = 255, message = "备注长度不能超过255")
    private String remark;

}
