package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 角色分页查询实体
 *
 * @author earthyzinc
 * @since 2022/6/3
 *
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class RolePageQuery extends BasePageQuery {

    @Schema(description="关键字(角色名称/角色编码)")
    private String keywords;
}
