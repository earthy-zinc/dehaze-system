package com.pei.dehaze.sdk.model.message;

import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class MessageTemplateQuery extends PageQuery {
    private String name;
    private String type;
    private Integer status;
}
