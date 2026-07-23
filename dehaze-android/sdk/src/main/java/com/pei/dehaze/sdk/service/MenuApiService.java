package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.menu.MenuForm;
import com.pei.dehaze.sdk.model.menu.MenuVO;
import retrofit2.Call;
import retrofit2.http.*;

import java.util.List;

/**
 * 菜单相关API服务接口
 */
public interface MenuApiService {
    // Menu APIs
    @GET("/api/v1/menus")
    Call<Result<List<MenuVO>>> getMenuList(@Query("keywords") String keywords);

    @GET("/api/v1/menus/options")
    Call<Result<List<Option>>> getMenuOptions();

    @GET("/api/v1/menus/{id}/form")
    Call<Result<MenuForm>> getMenuFormData(@Path("id") long id);

    @POST("/api/v1/menus")
    Call<Result<Void>> addMenu(@Body MenuForm data);

    @PUT("/api/v1/menus/{id}")
    Call<Result<Void>> updateMenu(@Path("id") long id, @Body MenuForm data);

    @DELETE("/api/v1/menus/{id}")
    Call<Result<Void>> deleteMenu(@Path("id") long id);

    @PATCH("/api/v1/menus/{id}")
    Call<Result<Void>> updateMenuVisible(@Path("id") long id, @Query("visible") int visible);
}
