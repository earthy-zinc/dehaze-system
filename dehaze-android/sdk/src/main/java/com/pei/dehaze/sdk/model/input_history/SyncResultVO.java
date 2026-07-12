package com.pei.dehaze.sdk.model.input_history;

import lombok.Data;

/**
 * 同步结果VO
 */
@Data
public class SyncResultVO {
    /** 已同步数量 */
    private int synced;
    /** 失败数量 */
    private int failed;
    /** 消息 */
    private String message;
}
