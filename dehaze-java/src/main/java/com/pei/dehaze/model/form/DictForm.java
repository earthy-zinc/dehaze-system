package com.pei.dehaze.model.form;


import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Schema(description = "字典表单对象")
@Data
public class DictForm {

    @Schema(description="字典ID")
    private Long id;

    @Schema(description="类型编码")
    @NotBlank(message = "字典类型编码不能为空")
    @Size(max = 50, message = "类型编码长度不能超过50")
    private String typeCode;

    @Schema(description="字典名称")
    @NotBlank(message = "字典名称不能为空")
    @Size(max = 50, message = "字典名称长度不能超过50")
    @Pattern(regexp = "(?s)^(?!.*javascript:)(?!.*<[a-zA-Z]).*$", message = "字典名称不能包含特殊字符")
    private String name;

    @Schema(description="字典值")
    @NotBlank(message = "字典值不能为空")
    @Size(max = 50, message = "字典值长度不能超过50")
    private String value;

    @Schema(description="状态(1:启用;0:禁用)")
    @Min(value = 0, message = "状态取值只能为0或1")
    @Max(value = 1, message = "状态取值只能为0或1")
    private Integer status;

    @Schema(description="排序")
    @Min(value = 1, message = "排序必须为正整数")
    private Integer sort;

    @Schema(description="是否默认(1:是;0:否)")
    @Min(value = 0, message = "是否默认取值只能为0或1")
    @Max(value = 1, message = "是否默认取值只能为0或1")
    private Integer defaulted;

    @Schema(description = "字典备注")
    private String remark;

}
