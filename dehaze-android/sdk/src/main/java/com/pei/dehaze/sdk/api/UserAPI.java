package com.pei.dehaze.sdk.api;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.EnableStatus;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.user.*;
import okhttp3.ResponseBody;
import retrofit2.Call;

import java.io.File;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 用户相关API接口封装
 */
public class UserAPI {

    private UserAPI() {
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
                        queryParams.getStatus() != null ? queryParams.getStatus().getValue() : null,
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
    public static void deleteByIds(List<Long> ids, ApiCallback<Void> callback) {
        String joined = ids.stream().map(String::valueOf).collect(Collectors.joining(","));
        Call<Result<Void>> call = DehazeSDK.getInstance().getUserApiService().deleteUsers(joined);
        call.enqueue(callback);
    }

    /**
     * 修改用户状态
     *
     * @param id       用户ID
     * @param status   状态
     * @param callback 回调函数
     */
    public static void updateStatus(long id, EnableStatus status, ApiCallback<Void> callback) {
        Call<Result<Void>> call = DehazeSDK.getInstance().getUserApiService().updateUserStatus(id, status.getValue());
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
        FileAPI.enqueueFileDownload(call, filePath, callback);
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
                        queryParams.getStatus() != null ? queryParams.getStatus().getValue() : null,
                        queryParams.getDeptId(), queryParams.getStartTime(), queryParams.getEndTime(),
                        "application/octet-stream");
        FileAPI.enqueueFileDownload(call, filePath, callback);
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
}
