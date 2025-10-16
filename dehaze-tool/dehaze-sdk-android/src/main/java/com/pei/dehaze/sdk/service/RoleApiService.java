package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.role.RoleForm;
import com.pei.dehaze.sdk.model.role.RolePageVO;
import retrofit2.Call;
import retrofit2.http.*;

import java.util.List;

/**
 * 角色相关API服务接口
 */
public interface RoleApiService {
    // Role APIs
    @GET("/api/v1/roles/page")
    Call<Result<PageResult<RolePageVO>>> getRolePage(@Query("pageNum") int pageNum,
                                                     @Query("pageSize") int pageSize,
                                                     @Query("keywords") String keywords);

    @GET("/api/v1/roles/options")
    Call<Result<List<Option>>> getRoleOptions(@Query("pageNum") int pageNum,
                                              @Query("pageSize") int pageSize,
                                              @Query("keywords") String keywords);

    @GET("/api/v1/roles/{id}/menuIds")
    Call<Result<List<Integer>>> getRoleMenuIds(@Path("id") int id);

    @PUT("/api/v1/roles/{id}/menus")
    Call<Result<Void>> updateRoleMenus(@Path("id") int id, @Body List<Integer> data);

    @GET("/api/v1/roles/{id}/form")
    Call<Result<RoleForm>> getRoleFormData(@Path("id") int id);

    @POST("/api/v1/roles")
    Call<Result<Void>> addRole(@Body RoleForm data);

    @PUT("/api/v1/roles/{id}")
    Call<Result<Void>> updateRole(@Path("id") int id, @Body RoleForm data);

    @DELETE("/api/v1/roles/{ids}")
    Call<Result<Void>> deleteRoles(@Path("ids") String ids);
}
