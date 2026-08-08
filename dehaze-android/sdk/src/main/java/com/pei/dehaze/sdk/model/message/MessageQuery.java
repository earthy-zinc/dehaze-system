package com.pei.dehaze.sdk.model.message;

import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class MessageQuery extends PageQuery {
    private String type;
    private Integer readStatus;
}
