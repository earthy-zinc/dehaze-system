package com.pei.dehaze.repository;

import android.os.Environment;

import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.file.FileInfo;

import java.io.File;

/**
 * 文件管理 Repository
 */
public class FileRepository {

    /**
     * 分页查询文件列表
     */
    public void getFiles(int pageNum, int pageSize, String keywords, RepositoryCallback<PageResult<FileInfo>> callback) {
        FileAPI.getFilePage(pageNum, pageSize, keywords, RepositoryAdapters.wrap(callback));
    }

    /**
     * 上传文件
     */
    public void uploadFile(File file, RepositoryCallback<FileInfo> callback) {
        FileAPI.upload(file, RepositoryAdapters.wrap(callback));
    }

    /**
     * 下载文件到应用专属下载目录（无需存储权限）
     *
     * @param objectName 对象存储名称
     * @param fileName   保存的文件名
     */
    public void downloadFile(String objectName, String fileName, RepositoryCallback<Void> callback) {
        FileAPI.downloadFile(objectName, RepositoryAdapters.wrap(new RepositoryCallback<okhttp3.ResponseBody>() {
            @Override
            public void onSuccess(okhttp3.ResponseBody data) {
                try {
                    File downloadDir = new File(Environment.getExternalStorageDirectory(), Environment.DIRECTORY_DOWNLOADS);
                    if (!downloadDir.exists()) {
                        downloadDir.mkdirs();
                    }
                    File saveFile = new File(downloadDir, fileName);
                    FileAPI.saveToFile(data, saveFile.getAbsolutePath());
                    callback.onSuccess(null);
                } catch (Exception e) {
                    callback.onError("文件保存失败: " + e.getMessage());
                }
            }

            @Override
            public void onError(String errorMessage) {
                callback.onError(errorMessage);
            }
        }));
    }

    /**
     * 删除文件
     */
    public void deleteFile(long fileId, RepositoryCallback<Void> callback) {
        FileAPI.delete(fileId, RepositoryAdapters.wrap(callback));
    }

    /**
     * 获取文件详情
     */
    public void getFileDetail(long fileId, RepositoryCallback<FileInfo> callback) {
        FileAPI.getFileDetail(fileId, RepositoryAdapters.wrap(callback));
    }
}
