package com.pei.dehaze.sdk.model.task;

import com.pei.dehaze.sdk.model.PageQuery;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 任务分页查询参数
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class TaskQuery extends PageQuery {
    /** 状态筛选（pending/processing/completed/failed/cancelled） */
    private String status;
    /** 类型筛选 */
    private String taskType;
}
