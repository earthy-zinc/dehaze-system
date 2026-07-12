package com.pei.dehaze.repository;

import android.os.Environment;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.network.ApiException;

import java.io.File;
import java.util.List;

/**
 * 文件管理 Repository
 */
public class FileRepository {

    /**
     * 文件列表回调
     */
    public interface FileListCallback {
        void onSuccess(List<FileInfo> files, long total);
        void onError(String code, String message);
    }

    /**
     * 单文件操作回调
     */
    public interface FileCallback {
        void onSuccess(FileInfo file);
        void onError(String code, String message);
    }

    /**
     * 无返回值操作回调（删除、下载等）
     */
    public interface ActionCallback {
        void onSuccess();
        void onError(String code, String message);
    }

    /**
     * 分页查询文件列表
     */
    public void getFiles(int pageNum, int pageSize, String keywords, FileListCallback callback) {
        FileAPI.getFilePage(pageNum, pageSize, keywords, new ApiCallback<PageResult<FileInfo>>() {
            @Override
            public void onSuccess(PageResult<FileInfo> data) {
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
     * 上传文件
     */
    public void uploadFile(File file, FileCallback callback) {
        FileAPI.upload(file, new ApiCallback<FileInfo>() {
            @Override
            public void onSuccess(FileInfo data) {
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
     * 下载文件到应用专属下载目录（无需存储权限）
     *
     * @param objectName 对象存储名称
     * @param fileName   保存的文件名
     */
    public void downloadFile(String objectName, String fileName, ActionCallback callback) {
        FileAPI.downloadFile(objectName, new ApiCallback<okhttp3.ResponseBody>() {
            @Override
            public void onSuccess(okhttp3.ResponseBody data) {
                try {
                    File downloadDir = new File(Environment.getExternalStorageDirectory(), Environment.DIRECTORY_DOWNLOADS);
                    if (!downloadDir.exists()) {
                        downloadDir.mkdirs();
                    }
                    File saveFile = new File(downloadDir, fileName);
                    FileAPI.saveToFile(data, saveFile.getAbsolutePath());
                    callback.onSuccess();
                } catch (Exception e) {
                    callback.onError("IO_ERROR", "文件保存失败: " + e.getMessage());
                }
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
     * 删除文件
     */
    public void deleteFile(long fileId, ActionCallback callback) {
        FileAPI.delete(fileId, new ApiCallback<Void>() {
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
     * 获取文件详情
     */
    public void getFileDetail(long fileId, FileCallback callback) {
        FileAPI.getFileDetail(fileId, new ApiCallback<FileInfo>() {
            @Override
            public void onSuccess(FileInfo data) {
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
}
