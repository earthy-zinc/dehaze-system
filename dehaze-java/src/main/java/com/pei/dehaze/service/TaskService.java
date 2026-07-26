package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.model.query.TaskQuery;
import com.pei.dehaze.model.vo.TaskVO;

/**
 * 任务服务接口
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
public interface TaskService extends IService<SysTask> {

    /**
     * 创建任务（统一入口）
     *
     * @param form            任务创建表单
     * @param idempotencyKey  客户端幂等键（可空），相同键返回已有任务
     * @return 任务VO
     */
    TaskVO createTask(ExportTaskCreateForm form, String idempotencyKey);

    /**
     * 查询任务状态
     *
     * @param taskId 任务ID
     * @return 任务VO
     */
    TaskVO getTaskStatus(String taskId);

    /**
     * 获取下载链接
     *
     * @param taskId 任务ID
     * @return 下载链接
     */
    String getDownloadUrl(String taskId);

    /**
     * 取消任务
     *
     * @param taskId 任务ID
     */
    void cancelTask(String taskId);

    /**
     * 重试失败的任务（重放入口）
     *
     * @param taskId 任务ID
     * @return 任务VO
     */
    TaskVO retryTask(String taskId);

    /**
     * 分页查询当前用户的任务列表
     *
     * @param pageNum  页码
     * @param pageSize 页大小
     * @return 分页结果
     */
    PageResult<TaskVO> listMyTasks(int pageNum, int pageSize);

    /**
     * 分页查询当前用户的任务列表（支持按任务类型、类别筛选）
     *
     * @param query 查询参数
     * @return 分页结果
     */
    PageResult<TaskVO> listMyTasks(TaskQuery query);

    /**
     * 任务执行完成后更新状态（供 GenericExportStrategy/GenericImportStrategy 调用）
     *
     * @param taskId     任务ID（UUID）
     * @param result     结果数据（导出为下载URL，导入为结果JSON）
     * @param expiresAt  过期时间
     */
    void updateTaskCompleted(String taskId, String result, java.time.LocalDateTime expiresAt);

    /**
     * 任务执行失败时更新状态
     *
     * @param taskId       任务ID（UUID）
     * @param errorMessage 错误信息
     */
    void updateTaskFailed(String taskId, String errorMessage);
}
