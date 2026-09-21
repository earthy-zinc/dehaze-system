package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.time.LocalDateTime;

@Schema(description = "角色表单对象")
@Data
public class RoleForm {

    @Schema(description="角色ID")
    private Long id;

    @Schema(description="角色名称")
    @NotBlank(message = "角色名称不能为空")
    @Size(max = 64, message = "角色名称长度不能超过64")
    @Pattern(regexp = "^(?!.*javascript:)(?!.*<[a-zA-Z]).*$", message = "角色名称不能包含特殊字符")
    private String name;

    @Schema(description="角色编码")
    @NotBlank(message = "角色编码不能为空")
    @Size(max = 32, message = "角色编码长度不能超过32")
    @Pattern(regexp = "^(?!.*javascript:)(?!.*<[a-zA-Z]).*$", message = "角色编码不能包含特殊字符")
    private String code;

    @Schema(description="排序")
    @Min(value = 0, message = "排序不能为负数")
    private Integer sort;

    @Schema(description="角色状态(1-正常；0-停用)")
    @Min(value = 0, message = "角色状态取值只能为0或1")
    @Max(value = 1, message = "角色状态取值只能为0或1")
    private Integer status;

    @Schema(description="数据权限(0-全部数据；1-部门及子部门数据；2-本部门数据；3-本人数据)")
    @Min(value = 0, message = "数据权限取值只能为0-3")
    @Max(value = 3, message = "数据权限取值只能为0-3")
    private Integer dataScope;

    @Schema(description="数据权限范围中文描述")
    private String dataScopeLabel;

    @Schema(description="创建时间")
    private LocalDateTime createTime;

    @Schema(description="更新时间")
    private LocalDateTime updateTime;

}
