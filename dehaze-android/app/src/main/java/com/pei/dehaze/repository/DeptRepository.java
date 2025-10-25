package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.DeptAPI;
import com.pei.dehaze.sdk.model.dept.DeptQuery;
import com.pei.dehaze.sdk.model.dept.DeptVO;

import java.util.List;

public class DeptRepository {
    
    public interface DeptCallback {
        void onSuccess(List<DeptVO> depts);
        void onError(String errorMessage);
    }
    
    public void getDepts(DeptCallback callback) {
        DeptQuery query = new DeptQuery();
        
        DeptAPI.getList(query, new com.pei.dehaze.sdk.ApiCallback<List<DeptVO>>() {
            @Override
            public void onSuccess(List<DeptVO> data) {
                callback.onSuccess(data);
            }
            
            @Override
            public void onError(int code, String message) {
                callback.onError("Error " + code + ": " + message);
            }
            
            @Override
            public void onFailure(com.pei.dehaze.sdk.network.ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }
}