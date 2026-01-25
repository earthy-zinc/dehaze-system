package com.pei.dehaze.common.constant;

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
}
