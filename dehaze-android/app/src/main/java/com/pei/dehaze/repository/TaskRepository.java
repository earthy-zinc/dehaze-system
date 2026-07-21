package com.pei.dehaze.repository;

import android.os.Environment;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.TaskAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.task.TaskCreateForm;
import com.pei.dehaze.sdk.model.task.TaskQuery;
import com.pei.dehaze.sdk.model.task.TaskVO;
import com.pei.dehaze.sdk.network.ApiException;

import java.io.File;
import java.util.List;

/**
 * 任务管理 Repository
 */
public class TaskRepository {

    /**
     * 任务列表回调
     */
    public interface TaskListCallback {
        void onSuccess(List<TaskVO> tasks, long total);
        void onError(String code, String message);
    }

    /**
     * 单任务操作回调
     */
    public interface TaskCallback {
        void onSuccess(TaskVO task);
        void onError(String code, String message);
    }

    /**
     * 无返回值操作回调（取消、删除、下载等）
     */
    public interface ActionCallback {
        void onSuccess();
        void onError(String code, String message);
    }

    /**
     * 分页查询任务列表
     */
    public void getTasks(TaskQuery query, TaskListCallback callback) {
        TaskAPI.getTaskPage(query, new ApiCallback<PageResult<TaskVO>>() {
            @Override
            public void onSuccess(PageResult<TaskVO> data) {
                callback.onSuccess(data.getList(), data.getTotal());
            }

            @Override
            public void onError(String code, String message) {
                callback.onError(code, message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getCode(), e.getMessage());
            }
        });
    }

    /**
     * 创建任务
     */
    public void createTask(TaskCreateForm form, TaskCallback callback) {
        TaskAPI.createTask(form, new ApiCallback<TaskVO>() {
            @Override
            public void onSuccess(TaskVO data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError(code, message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getCode(), e.getMessage());
            }
        });
    }

    /**
     * 查询任务详情
     */
    public void getTask(String taskId, TaskCallback callback) {
        TaskAPI.getTask(taskId, new ApiCallback<TaskVO>() {
            @Override
            public void onSuccess(TaskVO data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError(code, message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getCode(), e.getMessage());
            }
        });
    }

    /**
     * 取消任务
     */
    public void cancelTask(String taskId, ActionCallback callback) {
        TaskAPI.cancelTask(taskId, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess();
            }

            @Override
            public void onError(String code, String message) {
                callback.onError(code, message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getCode(), e.getMessage());
            }
        });
    }

    /**
     * 下载任务结果文件到下载目录
     *
     * @param taskId 任务ID
     */
    public void downloadTaskFile(String taskId, ActionCallback callback) {
        File downloadDir = new File(Environment.getExternalStorageDirectory(), Environment.DIRECTORY_DOWNLOADS);
        if (!downloadDir.exists()) {
            downloadDir.mkdirs();
        }
        String savePath = new File(downloadDir, "task_" + taskId + ".zip").getAbsolutePath();
        TaskAPI.downloadTaskFile(taskId, savePath, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess();
            }

            @Override
            public void onError(String code, String message) {
                callback.onError(code, message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getCode(), e.getMessage());
            }
        });
    }
}
