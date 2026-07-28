package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDate;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "会员分页查询对象")
public class MemberPageQuery extends BasePageQuery {

    @Schema(description = "关键字（用户名/昵称/手机号）")
    private String keywords;

    @Schema(description = "会员等级(level_0/level_1/level_2/level_3)")
    private String levelCode;

    @Schema(description = "会员状态(1:正常;0:冻结)")
    private Integer status;

    @Schema(description = "到期时间-开始")
    private LocalDate expireTimeStart;

    @Schema(description = "到期时间-结束")
    private LocalDate expireTimeEnd;

    @Schema(description = "成长值下限")
    private Long growthMin;

    @Schema(description = "成长值上限")
    private Long growthMax;
}
