package com.pei.dehaze.sdk.model.order;

import com.pei.dehaze.sdk.model.PageQuery;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class OrderQuery extends PageQuery {
    private String orderNo;
    private String keywords;
    private String status;
    private String payMethod;
    private Double amountMin;
    private Double amountMax;
    private String paidTimeStart;
    private String paidTimeEnd;
}
