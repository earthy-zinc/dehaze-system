package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.RoleAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.role.RolePageVO;
import com.pei.dehaze.sdk.model.role.RoleQuery;

import java.util.List;

public class RoleRepository {
    
    public interface RoleCallback {
        void onSuccess(List<RolePageVO> roles);
        void onError(String errorMessage);
    }
    
    public void getRoles(RoleCallback callback) {
        RoleQuery query = new RoleQuery();
        query.setPageNum(1);
        query.setPageSize(20);
        
        RoleAPI.getPage(query, new com.pei.dehaze.sdk.ApiCallback<PageResult<RolePageVO>>() {
            @Override
            public void onSuccess(PageResult<RolePageVO> data) {
                callback.onSuccess(data.getList());
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