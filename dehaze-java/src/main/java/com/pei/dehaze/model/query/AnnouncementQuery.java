package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "公告分页查询对象")
public class AnnouncementQuery extends BasePageQuery {

    @Schema(description = "公告标题（模糊）")
    private String title;

    @Schema(description = "公告类型")
    private String type;

    @Schema(description = "公告状态")
    private Integer status;
}
