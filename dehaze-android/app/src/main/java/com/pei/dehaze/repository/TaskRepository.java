package com.pei.dehaze.repository;

import android.os.Environment;

import com.pei.dehaze.sdk.api.TaskAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.task.TaskCreateForm;
import com.pei.dehaze.sdk.model.task.TaskQuery;
import com.pei.dehaze.sdk.model.task.TaskVO;

import java.io.File;

/**
 * 任务管理 Repository
 */
public class TaskRepository {

    /**
     * 分页查询任务列表
     */
    public void getTasks(TaskQuery query, RepositoryCallback<PageResult<TaskVO>> callback) {
        TaskAPI.getTaskPage(query, RepositoryAdapters.wrap(callback));
    }

    /**
     * 创建任务
     */
    public void createTask(TaskCreateForm form, RepositoryCallback<TaskVO> callback) {
        TaskAPI.createTask(form, RepositoryAdapters.wrap(callback));
    }

    /**
     * 查询任务详情
     */
    public void getTask(String taskId, RepositoryCallback<TaskVO> callback) {
        TaskAPI.getTask(taskId, RepositoryAdapters.wrap(callback));
    }

    /**
     * 取消任务
     */
    public void cancelTask(String taskId, RepositoryCallback<Void> callback) {
        TaskAPI.cancelTask(taskId, RepositoryAdapters.wrap(callback));
    }

    /**
     * 下载任务结果文件到下载目录
     *
     * @param taskId 任务ID
     */
    public void downloadTaskFile(String taskId, RepositoryCallback<Void> callback) {
        File downloadDir = new File(Environment.getExternalStorageDirectory(), Environment.DIRECTORY_DOWNLOADS);
        if (!downloadDir.exists()) {
            downloadDir.mkdirs();
        }
        String savePath = new File(downloadDir, "task_" + taskId + ".zip").getAbsolutePath();
        TaskAPI.downloadTaskFile(taskId, savePath, RepositoryAdapters.wrap(callback));
    }
}
