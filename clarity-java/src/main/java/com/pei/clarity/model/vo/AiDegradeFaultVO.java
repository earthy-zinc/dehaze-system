package com.pei.clarity.model.vo;

import java.util.List;
import lombok.Data;

/**
 * 降级与 Key 故障统计
 *
 * @author clarity
 */
@Data
public class AiDegradeFaultVO {

    private List<AiDowngradeVO> downgradeFrequency;

    private Integer keyFailoverCount;
}
