package com.pei.clarity.sdk.model.message;

import com.pei.clarity.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class AnnouncementQuery extends PageQuery {
    private String title;
    private String type;
    private Integer status;
}
