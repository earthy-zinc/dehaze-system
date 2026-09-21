package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;

/**
 * 菜单查询对象
 *
 * @author earthyzinc
 * @since 2022/10/28
 */
@Schema(description ="菜单查询对象")
@Data
public class MenuQuery {

    @Schema(description="关键字(菜单名称)")
    private String keywords;

    @Schema(description="权限标识(模糊匹配)")
    private String perm;

    @Schema(description="路由地址(模糊匹配)")
    private String path;

    @Schema(description="菜单类型(1-菜单；2-目录；3-外链；4-按钮)")
    @Min(value = 1, message = "菜单类型不合法")
    @Max(value = 4, message = "菜单类型不合法")
    private Integer type;

    @Schema(description="显示状态(1:显示;0:隐藏)")
    @Min(value = 0, message = "显示状态不合法")
    @Max(value = 1, message = "显示状态不合法")
    private Integer visible;

}
