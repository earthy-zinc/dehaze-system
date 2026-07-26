package com.pei.dehaze.service.importexport.model;

import lombok.Data;

import java.io.OutputStream;
import java.util.List;
import java.util.Map;

/**
 * 导出上下文
 * <p>封装导出执行期间所需的全部信息，传递给 {@link com.pei.dehaze.service.importexport.ExportHandler}
 */
@Data
public class ExportContext {

    /** 任务ID */
    private String taskId;

    /** 模块标识（如 user/role/dept） */
    private String module;

    /** 文件格式：excel / csv */
    private String format;

    /** 选中的导出字段（为 null 表示导出全部非隐藏字段） */
    private List<String> selectedFields;

    /** 查询参数（来自列表页筛选条件） */
    private Map<String, Object> queryParams;

    /** 输出流（由调用方提供，Handler 负责写入） */
    private OutputStream outputStream;

    /** 总数据量（用于进度计算） */
    private long totalCount;

    /** 异步任务标识（同步导出时为 false） */
    private boolean async;
}
