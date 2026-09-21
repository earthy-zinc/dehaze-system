package com.pei.dehaze.model.vo;

import lombok.Data;

import java.util.List;

/**
 * 降级与 Key 故障统计
 *
 * @author dehaze
 */
@Data
public class AiDegradeFaultVO {

    private List<AiDowngradeVO> downgradeFrequency;

    private Integer keyFailoverCount;
}
