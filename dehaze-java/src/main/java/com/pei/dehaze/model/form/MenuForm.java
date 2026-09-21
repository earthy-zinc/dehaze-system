package com.pei.dehaze.model.form;

import com.pei.dehaze.common.enums.MenuTypeEnum;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Schema(description = "菜单表单对象")
@Data
public class MenuForm {

    @Schema(description = "菜单ID")
    private Long id;

    @Schema(description = "父菜单ID")
    private Long parentId;

    @Schema(description = "菜单名称")
    @NotBlank(message = "菜单名称不能为空")
    @Size(max = 64, message = "菜单名称长度不能超过64")
    // (?s) 允许名称原样存储 CRLF 等换行类脏语料（与 python validate_no_xss 口径一致，仅挡 XSS 特征）
    @Pattern(regexp = "^(?s)(?!.*javascript:)(?!.*<[a-zA-Z]).*$", message = "菜单名称不能包含特殊字符")
    private String name;

    @Schema(description = "菜单类型(1-菜单；2-目录；3-外链；4-按钮权限)")
    @NotNull(message = "菜单类型不能为空")
    private MenuTypeEnum type;

    @Schema(description = "路由路径")
    private String path;

    @Schema(description = "组件路径(vue页面完整路径，省略.vue后缀)")
    private String component;

    @Schema(description = "权限标识")
    private String perm;

    @Schema(description = "显示状态(1:显示;0:隐藏)")
    private Integer visible;

    @Schema(description = "排序(数字越小排名越靠前)")
    @Min(value = 0, message = "排序不能为负数")
    private Integer sort;

    @Schema(description = "菜单图标")
    private String icon;

    @Schema(description = "跳转路径")
    private String redirect;

    @Schema(description = "【菜单】是否开启页面缓存", example = "1")
    private Integer keepAlive;

    @Schema(description = "【目录】只有一个子路由是否始终显示", example = "1")
    private Integer alwaysShow;


}
