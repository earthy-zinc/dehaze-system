package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.network.ApiException;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;

/**
 * 文件相关API接口封装
 */
public class FileAPI {

    private FileAPI() {
    }

    /**
     * 将 ResponseBody 流式写入本地文件
     *
     * @param body     响应体
     * @param savePath 保存路径
     * @throws IOException IO异常
     */
    public static void saveToFile(ResponseBody body, String savePath) throws IOException {
        File outputFile = new File(savePath);
        File parentDir = outputFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        try (InputStream inputStream = body.byteStream();
             FileOutputStream outputStream = new FileOutputStream(outputFile)) {
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
        }
    }

    /**
     * 分页查询文件列表
     *
     * @param pageNum  页码
     * @param pageSize 每页数量
     * @param keywords 关键词
     * @param callback 回调函数
     */
    public static void getFilePage(int pageNum, int pageSize, String keywords,
                                    ApiCallback<PageResult<FileInfo>> callback) {
        Call<Result<PageResult<FileInfo>>> call = DehazeSDK.getInstance().getFileApiService()
                .getFilePage(pageNum, pageSize, keywords);
        call.enqueue(callback);
    }

    /**
     * 文件下载
     *
     * @param objectName 对象存储名称
     * @param callback   回调函数
     */
    public static void downloadFile(String objectName, ApiCallback<ResponseBody> callback) {
        Call<ResponseBody> call = DehazeSDK.getInstance().getFileApiService().downloadFile(objectName);
        call.enqueue(new retrofit2.Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, retrofit2.Response<ResponseBody> response) {
                if (response.isSuccessful() && response.body() != null) {
                    callback.onSuccess(response.body());
                } else {
                    callback.onFailure(new ApiException(response.code(), response.message()));
                }
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                callback.onFailure(new ApiException(0, t.getMessage()));
            }
        });
    }

    /**
     * 获取文件详情
     *
     * @param fileId   文件ID
     * @param callback 回调函数
     */
    public static void getFileDetail(long fileId, ApiCallback<FileInfo> callback) {
        Call<Result<FileInfo>> call = DehazeSDK.getInstance().getFileApiService().getFileDetail(fileId);
        call.enqueue(callback);
    }

    /**
     * 上传文件
     *
     * @param file     文件
     * @param callback 回调函数
     */
    public static void upload(File file, ApiCallback<FileInfo> callback) {
        RequestBody requestFile = RequestBody.create(MediaType.parse("*/*"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", file.getName(), requestFile);

        Call<Result<FileInfo>> call = DehazeSDK.getInstance().getFileApiService().uploadFile(body);
        call.enqueue(callback);
    }

    /**
     * 删除文件
     *
     * @param fileId   文件ID
     * @param callback 回调函数
     */
    public static void delete(long fileId, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getFileApiService().deleteFile(fileId);
        call.enqueue(callback);
    }

    /**
     * 通用文件下载：将 ResponseBody 流保存到本地路径。
     * 供 UserAPI.downloadTemplate / UserAPI.export / TaskAPI.downloadTaskFile 等复用。
     *
     * @param call      已构造的下载 Call（返回 ResponseBody）
     * @param filePath  本地保存路径
     * @param callback  回调（成功时 data 为 null）
     */
    public static void enqueueFileDownload(Call<ResponseBody> call, String filePath, ApiCallback<Void> callback) {
        call.enqueue(new retrofit2.Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, retrofit2.Response<ResponseBody> response) {
                if (response.isSuccessful() && response.body() != null) {
                    try {
                        FileAPI.saveToFile(response.body(), filePath);
                        callback.onSuccess(null);
                    } catch (IOException e) {
                        callback.onFailure(new ApiException(-1, "文件保存失败: " + e.getMessage()));
                    }
                } else {
                    callback.onFailure(new ApiException(response.code(), response.message()));
                }
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                callback.onFailure(new ApiException(-1, t.getMessage()));
            }
        });
    }
}
