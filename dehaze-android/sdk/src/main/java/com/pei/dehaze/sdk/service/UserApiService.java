package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.user.UserForm;
import com.pei.dehaze.sdk.model.user.UserInfo;
import com.pei.dehaze.sdk.model.user.UserPageVO;
import okhttp3.MultipartBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.*;

/**
 * 用户相关API服务接口
 */
public interface UserApiService {
    /**
     * 登录成功后获取用户信息（昵称、头像、权限集合和角色集合）
     */
    @GET("/api/v1/auth/me")
    Call<Result<UserInfo>> getUserDetails();

    /**
     * 获取用户分页列表
     */
    @GET("/api/v1/users/page")
    Call<Result<PageResult<UserPageVO>>> getUserPage(@Query("pageNum") int pageNum,
                                                     @Query("pageSize") int pageSize,
                                                     @Query("keywords") String keywords,
                                                     @Query("status") Integer status,
                                                     @Query("deptId") Integer deptId,
                                                     @Query("startTime") String startTime,
                                                     @Query("endTime") String endTime);

    /**
     * 获取用户表单详情
     */
    @GET("/api/v1/users/{userId}/form")
    Call<Result<UserForm>> getUserFormData(@Path("userId") int userId);

    /**
     * 添加用户
     */
    @POST("/api/v1/users")
    Call<Result<Void>> addUser(@Body UserForm data);

    /**
     * 修改用户
     */
    @PUT("/api/v1/users/{id}")
    Call<Result<Void>> updateUser(@Path("id") int id, @Body UserForm data);

    /**
     * 修改用户密码
     */
    @PATCH("/api/v1/users/{id}/password")
    Call<Result<Void>> updateUserPassword(@Path("id") int id, @Query("password") String password);

    /**
     * 修改用户状态
     */
    @PATCH("/api/v1/users/{id}/status")
    Call<Result<Void>> updateUserStatus(@Path("id") long id, @Query("status") int status);

    /**
     * 删除用户
     */
    @DELETE("/api/v1/users/{ids}")
    Call<Result<Void>> deleteUsers(@Path("ids") String ids);

    /**
     * 下载用户导入模板
     */
    @Streaming
    @GET("/api/v1/users/template")
    Call<ResponseBody> downloadUserTemplate(@Header("Content-Type") String contentType);

    /**
     * 导出用户
     */
    @Streaming
    @GET("/api/v1/users/_export")
    Call<ResponseBody> exportUsers(@Query("pageNum") int pageNum,
                                   @Query("pageSize") int pageSize,
                                   @Query("keywords") String keywords,
                                   @Query("status") Integer status,
                                   @Query("deptId") Integer deptId,
                                   @Query("startTime") String startTime,
                                   @Query("endTime") String endTime,
                                   @Header("Content-Type") String contentType);

    /**
     * 导入用户
     */
    @Multipart
    @POST("/api/v1/users/_import")
    Call<Result<Void>> importUsers(@Query("deptId") int deptId, @Part MultipartBody.Part file);
}
