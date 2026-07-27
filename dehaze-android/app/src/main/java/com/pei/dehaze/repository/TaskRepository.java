package com.pei.dehaze.repository;

import android.os.Environment;

import com.pei.dehaze.sdk.api.TaskAPI;

import java.io.File;

/**
 * 任务管理 Repository
 */
public class TaskRepository {

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
