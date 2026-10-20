package com.pei.clarity.model.vo;

import java.time.OffsetDateTime;
import java.util.List;
import lombok.Data;

/**
 * Cron 解释与下次执行时间预览
 *
 * @author clarity
 */
@Data
public class AiNextTimesVO {

    private String description;

    private List<OffsetDateTime> nextTimes;
}
