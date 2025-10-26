package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.user.*;
import okhttp3.ResponseBody;
import retrofit2.Call;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * 用户相关API接口封装
 */
public class UserAPI {

    /**
     * 获取当前用户信息 (旧接口)
     *
     * @param callback 回调函数
     */
    public static void getCurrentUserInfo(ApiCallback<UserInfoResponse> callback) {
        Call<Result<UserInfoResponse>> call = DehazeSDK.getInstance().getUserApiService().getUserInfo();
        call.enqueue(callback);
    }

    /**
     * 登录成功后获取用户信息（昵称、头像、权限集合和角色集合）
     *
     * @param callback 回调函数
     */
    public static void getInfo(ApiCallback<UserInfo> callback) {
        Call<Result<UserInfo>> call = DehazeSDK.getInstance().getUserApiService().getUserDetails();
        call.enqueue(callback);
    }

    /**
     * 获取用户分页列表
     *
     * @param queryParams 查询参数
     * @param callback    回调函数
     */
    public static void getPage(UserQuery queryParams, ApiCallback<PageResult<UserPageVO>> callback) {
        Call<Result<PageResult<UserPageVO>>> call = DehazeSDK
                .getInstance()
                .getUserApiService()
                .getUserPage(
                        queryParams.getPageNum(),
                        queryParams.getPageSize(),
                        queryParams.getKeywords(),
                        queryParams.getStatus(),
                        queryParams.getDeptId(),
                        queryParams.getStartTime(),
                        queryParams.getEndTime()
                );
        call.enqueue(callback);
    }

    /**
     * 获取用户表单详情
     *
     * @param userId   用户ID
     * @param callback 回调函数
     */
    public static void getFormData(int userId, ApiCallback<UserForm> callback) {
        Call<Result<UserForm>> call = DehazeSDK.getInstance().getUserApiService().getUserFormData(userId);
        call.enqueue(callback);
    }

    /**
     * 添加用户
     *
     * @param data     用户表单数据
     * @param callback 回调函数
     */
    public static void add(UserForm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getUserApiService().addUser(data);
        call.enqueue(callback);
    }

    /**
     * 修改用户
     *
     * @param id       用户ID
     * @param data     用户表单数据
     * @param callback 回调函数
     */
    public static void update(int id, UserForm data, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getUserApiService().updateUser(id, data);
        call.enqueue(callback);
    }

    /**
     * 修改用户密码
     *
     * @param id       用户ID
     * @param password 新密码
     * @param callback 回调函数
     */
    public static void updatePassword(int id, String password, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getUserApiService().updateUserPassword(id, password);
        call.enqueue(callback);
    }

    /**
     * 删除用户
     *
     * @param ids      用户ID列表
     * @param callback 回调函数
     */
    public static void deleteByIds(String ids, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getUserApiService().deleteUsers(ids);
        call.enqueue(callback);
    }

    /**
     * 下载用户导入模板
     *
     * @param filePath 保存文件路径
     * @param callback 回调函数
     */
    public static void downloadTemplate(String filePath, ApiCallback<Void> callback) {
        Call<ResponseBody> call = DehazeSDK.getInstance().getUserApiService().downloadUserTemplate("application/octet-stream");
        call.enqueue(new retrofit2.Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, retrofit2.Response<ResponseBody> response) {
                if (response.isSuccessful() && response.body() != null) {
                    // 保存文件
                    try {
                        saveToFile(response.body(), filePath);
                        callback.onSuccess(null);
                    } catch (Exception e) {
                        callback.onFailure(new com.pei.dehaze.sdk.network.ApiException(-1, "文件保存失败: " + e.getMessage()));
                    }
                } else {
                    callback.onFailure(new com.pei.dehaze.sdk.network.ApiException(response.code(), response.message()));
                }
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                callback.onFailure(new com.pei.dehaze.sdk.network.ApiException(-1, t.getMessage()));
            }
        });
    }

    /**
     * 导出用户
     *
     * @param queryParams 查询参数
     * @param filePath    保存文件路径
     * @param callback    回调函数
     */
    public static void export(UserQuery queryParams, String filePath, ApiCallback<Void> callback) {
        Call<ResponseBody> call = DehazeSDK.getInstance().getUserApiService()
                .exportUsers(queryParams.getPageNum(), queryParams.getPageSize(), queryParams.getKeywords(),
                        queryParams.getStatus(), queryParams.getDeptId(), queryParams.getStartTime(), queryParams.getEndTime(),
                        "application/octet-stream");
        call.enqueue(new retrofit2.Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, retrofit2.Response<ResponseBody> response) {
                if (response.isSuccessful() && response.body() != null) {
                    // 保存文件
                    try {
                        saveToFile(response.body(), filePath);
                        callback.onSuccess(null);
                    } catch (Exception e) {
                        callback.onFailure(new com.pei.dehaze.sdk.network.ApiException(-1, "文件保存失败: " + e.getMessage()));
                    }
                } else {
                    callback.onFailure(new com.pei.dehaze.sdk.network.ApiException(response.code(), response.message()));
                }
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                callback.onFailure(new com.pei.dehaze.sdk.network.ApiException(-1, t.getMessage()));
            }
        });
    }

    /**
     * 导入用户
     *
     * @param deptId   部门ID
     * @param file     文件
     * @param callback 回调函数
     */
    public static void importUsers(int deptId, File file, ApiCallback<Void> callback) {
        // 创建MultipartBody.Part
        okhttp3.RequestBody requestFile = okhttp3.RequestBody.create(okhttp3.MediaType.parse("multipart/form-data"), file);
        okhttp3.MultipartBody.Part body = okhttp3.MultipartBody.Part.createFormData("file", file.getName(), requestFile);

        Call<Result<Void>> call = DehazeSDK.getInstance().getUserApiService().importUsers(deptId, body);
        call.enqueue(callback);
    }

    /**
     * 保存响应体到文件
     *
     * @param body     响应体
     * @param filePath 文件路径
     * @throws Exception 保存异常
     */
    private static void saveToFile(ResponseBody body, String filePath) throws Exception {
        InputStream inputStream = null;
        OutputStream outputStream = null;
        try {
            inputStream = body.byteStream();
            outputStream = new FileOutputStream(filePath);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            outputStream.flush();
        } finally {
            if (inputStream != null) {
                inputStream.close();
            }
            if (outputStream != null) {
                outputStream.close();
            }
        }
    }
}
