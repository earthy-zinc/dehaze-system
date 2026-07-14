package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
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
    PageResult<TaskVO> listMyTasks(Integer pageNum, Integer pageSize);
}
