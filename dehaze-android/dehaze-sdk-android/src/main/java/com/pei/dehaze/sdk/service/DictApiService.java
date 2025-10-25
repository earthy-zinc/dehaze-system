package com.pei.dehaze.sdk.service;

import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.Result;
import com.pei.dehaze.sdk.model.dict.*;
import retrofit2.Call;
import retrofit2.http.*;

import java.util.List;

/**
 * 字典相关API服务接口
 */
public interface DictApiService {
    // Dict APIs
    @GET("/api/v1/dict/types/page")
    Call<Result<PageResult<DictTypePageVO>>> getDictTypePage(@Query("pageNum") int pageNum,
                                                     @Query("pageSize") int pageSize,
                                                     @Query("keywords") String keywords);

    @GET("/api/v1/dict/types/{id}/form")
    Call<Result<DictTypeForm>> getDictTypeFormData(@Path("id") int id);

    @POST("/api/v1/dict/types")
    Call<Result<Void>> addDictType(@Body DictTypeForm data);

    @PUT("/api/v1/dict/types/{id}")
    Call<Result<Void>> updateDictType(@Path("id") int id, @Body DictTypeForm data);

    @DELETE("/api/v1/dict/types/{ids}")
    Call<Result<Void>> deleteDictTypes(@Path("ids") String ids);

    @GET("/api/v1/dict/{typeCode}/options")
    Call<Result<List<Option>>> getDictOptions(@Path("typeCode") String typeCode);

    @GET("/api/v1/dict/page")
    Call<Result<PageResult<DictPageVO>>> getDictPage(@Query("pageNum") int pageNum,
                                                     @Query("pageSize") int pageSize,
                                                     @Query("name") String name,
                                                     @Query("typeCode") String typeCode);

    @GET("/api/v1/dict/{id}/form")
    Call<Result<DictForm>> getDictFormData(@Path("id") int id);

    @POST("/api/v1/dict")
    Call<Result<Void>> addDict(@Body DictForm data);

    @PUT("/api/v1/dict/{id}")
    Call<Result<Void>> updateDict(@Path("id") int id, @Body DictForm data);

    @DELETE("/api/v1/dict/{ids}")
    Call<Result<Void>> deleteDicts(@Path("ids") String ids);
}
