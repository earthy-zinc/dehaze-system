package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.dept.DeptForm;
import com.pei.dehaze.sdk.model.dept.DeptVO;
import retrofit2.Call;
import retrofit2.http.*;

import java.util.List;

/**
 * 部门相关API服务接口
 */
public interface DeptApiService {
    // Dept APIs
    @GET("/api/v1/depts")
    Call<Result<List<DeptVO>>> getDeptList(@Query("keywords") String keywords, @Query("status") Integer status);

    @GET("/api/v1/depts/options")
    Call<Result<List<Option>>> getDeptOptions();

    @GET("/api/v1/depts/{id}/form")
    Call<Result<DeptForm>> getDeptFormData(@Path("id") int id);

    @POST("/api/v1/depts")
    Call<Result<Void>> addDept(@Body DeptForm data);

    @PUT("/api/v1/depts/{id}")
    Call<Result<Void>> updateDept(@Path("id") int id, @Body DeptForm data);

    @DELETE("/api/v1/depts/{ids}")
    Call<Result<Void>> deleteDepts(@Path("ids") String ids);
}
