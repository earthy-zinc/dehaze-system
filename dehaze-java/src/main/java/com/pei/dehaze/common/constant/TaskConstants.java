package com.pei.dehaze.common.constant;

import java.util.Set;

/**
 * 任务系统常量
 */
public final class TaskConstants {

    private TaskConstants() {}
    
    // ========== Redis Key 前缀 ==========
    /** 任务状态缓存前缀 */
    public static final String TASK_CACHE_PREFIX = "task:";
    /** 任务取消标识前缀 */
    public static final String TASK_CANCEL_PREFIX = "task:cancel:";
    /** 幂等键缓存前缀 */
    public static final String IDEMPOTENCY_KEY_PREFIX = "idempotency:task:";
    /** 幂等键缓存过期时间：24小时 */
    public static final long IDEMPOTENCY_KEY_EXPIRE_SECONDS = 24 * 60 * 60;

    // ========== WebSocket Pub/Sub ==========
    /** WebSocket 跨实例广播频道（对齐 Python dehaze:ws:broadcast） */
    public static final String WS_CHANNEL = "dehaze:ws:broadcast";
    
    // ========== 过期时间（秒） ==========
    /** 任务缓存过期时间：24小时 */
    public static final long TASK_EXPIRE_SECONDS = 24 * 60 * 60;
    /** 取消标识过期时间：5分钟 */
    public static final long CANCEL_FLAG_EXPIRE_SECONDS = 5 * 60;
    
    // ========== 任务类型（新规范） ==========
    /** 数据集导出 */
    public static final String TYPE_DATASET_EXPORT = "dataset_export";
    /** 数据项下载 */
    public static final String TYPE_ITEM_DOWNLOAD = "item_download";
    /** 批量下载 */
    public static final String TYPE_BATCH_DOWNLOAD = "batch_download";
    /** 自定义导出 */
    public static final String TYPE_CUSTOM_EXPORT = "custom_export";
    
    // ========== 任务状态 ==========
    /** 待处理 */
    public static final String STATUS_PENDING = "PENDING";
    /** 处理中 */
    public static final String STATUS_PROCESSING = "PROCESSING";
    /** 已完成 */
    public static final String STATUS_COMPLETED = "COMPLETED";
    /** 已失败 */
    public static final String STATUS_FAILED = "FAILED";
    /** 已取消 */
    public static final String STATUS_CANCELLED = "CANCELLED";

    // ========== 终态集合 ==========
    /** 任务终态集合（已完成 / 已失败 / 已取消），用于幂等检查 */
    public static final Set<String> TERMINAL_STATUSES = Set.of(
            STATUS_COMPLETED,
            STATUS_FAILED,
            STATUS_CANCELLED
    );
}
