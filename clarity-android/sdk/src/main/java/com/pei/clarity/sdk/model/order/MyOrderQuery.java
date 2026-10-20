package com.pei.clarity.sdk.model.order;

import com.pei.clarity.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class MyOrderQuery extends PageQuery {
    private String status;
}
