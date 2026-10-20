package com.pei.clarity.model.query;

import com.pei.clarity.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 权限分页查询对象
 *
 * @author earthyzinc
 * @since 2022/1/14 22:22
 */
@Data
@Schema
@EqualsAndHashCode(callSuper = true)
public class PermPageQuery extends BasePageQuery {

    @Schema(description = "权限名称")
    private String name;

    @Schema(description = "菜单ID")
    private Long menuId;
}
