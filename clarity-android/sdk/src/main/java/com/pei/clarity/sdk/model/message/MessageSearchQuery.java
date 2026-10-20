package com.pei.clarity.sdk.model.message;

import com.pei.clarity.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class MessageSearchQuery extends PageQuery {
    private String keyword;
}
