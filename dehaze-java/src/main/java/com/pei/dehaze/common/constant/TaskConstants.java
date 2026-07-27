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
    /** 导出/导入结果文件有效期：7天 */
    public static final long RESULT_FILE_EXPIRE_DAYS = 7;

    // ========== 任务类别 ==========
    /** 导入任务 */
    public static final String CATEGORY_IMPORT = "import";
    /** 导出任务 */
    public static final String CATEGORY_EXPORT = "export";

    // ========== 任务类型 - 通用导出 ==========
    /** 数据集导出（ZIP 归档，整合旧 item_download/batch_download/custom_export） */
    public static final String TYPE_DATASET_EXPORT = "dataset_export";
    public static final String TYPE_USER_EXPORT = "user_export";
    public static final String TYPE_ROLE_EXPORT = "role_export";
    public static final String TYPE_DEPT_EXPORT = "dept_export";
    public static final String TYPE_MENU_EXPORT = "menu_export";
    public static final String TYPE_DICT_EXPORT = "dict_export";
    public static final String TYPE_ALGORITHM_EXPORT = "algorithm_export";

    // ========== 任务类型 - 通用导入 ==========
    public static final String TYPE_USER_IMPORT = "user_import";
    public static final String TYPE_ROLE_IMPORT = "role_import";
    public static final String TYPE_DEPT_IMPORT = "dept_import";
    public static final String TYPE_MENU_IMPORT = "menu_import";
    public static final String TYPE_DICT_IMPORT = "dict_import";
    public static final String TYPE_ALGORITHM_IMPORT = "algorithm_import";

    // ========== 限制 ==========
    /** 单次导入/导出最大行数 */
    public static final int MAX_ROWS = 100_000;
    /** 单次导入文件最大大小（字节）：20MB */
    public static final long MAX_IMPORT_FILE_SIZE = 20L * 1024 * 1024;
    /** 同步导出/导入阈值（行数），超过则走异步任务 */
    public static final int SYNC_THRESHOLD = 1000;
    /** 分批查询/写入批次大小 */
    public static final int BATCH_SIZE = 1000;
    /** 导入批量插入事务粒度 */
    public static final int IMPORT_BATCH_TX_SIZE = 500;
    /** 单用户最大并发任务数 */
    public static final int MAX_CONCURRENT_PER_USER = 5;

    // ========== 任务状态 ==========
    /** 待处理 */
    public static final int STATUS_PENDING = 1;
    /** 处理中 */
    public static final int STATUS_PROCESSING = 2;
    /** 已完成 */
    public static final int STATUS_COMPLETED = 3;
    /** 已失败 */
    public static final int STATUS_FAILED = 4;
    /** 已取消 */
    public static final int STATUS_CANCELLED = 5;

    // ========== 终态集合 ==========
    /** 任务终态集合（已完成 / 已失败 / 已取消），用于幂等检查 */
    public static final Set<Integer> TERMINAL_STATUSES = Set.of(
            STATUS_COMPLETED,
            STATUS_FAILED,
            STATUS_CANCELLED
    );

    // ========== 模块标识 ==========
    public static final String MODULE_USER = "user";
    public static final String MODULE_ROLE = "role";
    public static final String MODULE_DEPT = "dept";
    public static final String MODULE_MENU = "menu";
    public static final String MODULE_DICT = "dict";
    public static final String MODULE_DATASET = "dataset";
    public static final String MODULE_ALGORITHM = "algorithm";

    /**
     * 根据任务类型判断任务类别
     */
    public static String getCategoryByType(String taskType) {
        if (taskType == null) {
            return null;
        }
        if (taskType.endsWith("_import")) {
            return CATEGORY_IMPORT;
        }
        if (taskType.endsWith("_export")) {
            return CATEGORY_EXPORT;
        }
        return null;
    }

    /**
     * 根据任务类型解析模块名（如 user_export -> user）
     */
    public static String getModuleByType(String taskType) {
        if (taskType == null) {
            return null;
        }
        if (taskType.endsWith("_import")) {
            return taskType.substring(0, taskType.length() - 7);
        }
        if (taskType.endsWith("_export")) {
            return taskType.substring(0, taskType.length() - 7);
        }
        return null;
    }
}
