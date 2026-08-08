package com.pei.dehaze.sdk.model.message;

import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class MessageSearchQuery extends PageQuery {
    private String keyword;
}
