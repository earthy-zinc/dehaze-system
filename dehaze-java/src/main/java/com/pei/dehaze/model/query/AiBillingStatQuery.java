package com.pei.dehaze.model.query;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/** 管理员计费统计查询参数 */
@Schema(description = "管理员计费统计查询参数")
@Data
public class AiBillingStatQuery {

    private Long userId;

    private String modelId;

    private String billType;

    private String dateStart;

    private String dateEnd;

    /** user/model/billType/day */
    private String groupBy = "model";
}
