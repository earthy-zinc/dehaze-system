package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.file.ImageFileInfo;

import java.io.File;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;

/**
 * 文件相关API接口封装
 */
public class FileAPI {

    /**
     * 文件上传检查
     *
     * @param md5      文件md5
     * @param callback 回调函数
     */
    public static void uploadCheck(String md5, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getFileApiService().checkFileUpload(md5);
        call.enqueue(callback);
    }

    /**
     * 上传文件
     *
     * @param file     文件
     * @param modelId  模型ID
     * @param callback 回调函数
     */
    public static void upload(File file, Integer modelId, ApiCallback<FileInfo> callback) {
        // 创建MultipartBody.Part
        RequestBody requestFile = RequestBody.create(MediaType.parse("*/*"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", file.getName(), requestFile);
        
        Call<Result<FileInfo>> call = DehazeSDK.getInstance().getFileApiService().uploadFile(body, modelId);
        call.enqueue(callback);
    }

    /**
     * 上传图片文件
     *
     * @param file     图片文件
     * @param modelId  模型ID
     * @param callback 回调函数
     */
    public static void uploadImage(File file, Integer modelId, ApiCallback<ImageFileInfo> callback) {
        // 创建MultipartBody.Part
        RequestBody requestFile = RequestBody.create(MediaType.parse("image/*"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", file.getName(), requestFile);
        
        Call<Result<ImageFileInfo>> call = DehazeSDK.getInstance().getFileApiService().uploadImageFile(body, modelId);
        call.enqueue(callback);
    }

    /**
     * 删除文件
     *
     * @param filePath 文件完整路径
     * @param callback 回调函数
     */
    public static void deleteByPath(String filePath, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getFileApiService().deleteFileByPath(filePath);
        call.enqueue(callback);
    }
}