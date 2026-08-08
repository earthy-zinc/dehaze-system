package com.pei.dehaze.sdk.model.message;

import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class AnnouncementQuery extends PageQuery {
    private String title;
    private String type;
    private Integer status;
}
