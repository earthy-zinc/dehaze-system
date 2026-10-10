package com.pei.dehaze.model.vo;

import java.time.OffsetDateTime;
import java.util.List;
import lombok.Data;

/**
 * Cron 解释与下次执行时间预览
 *
 * @author dehaze
 */
@Data
public class AiNextTimesVO {

    private String description;

    private List<OffsetDateTime> nextTimes;
}
