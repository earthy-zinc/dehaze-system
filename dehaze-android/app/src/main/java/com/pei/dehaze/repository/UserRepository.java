package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.UserAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.user.UserPageVO;
import com.pei.dehaze.sdk.model.user.UserQuery;

import java.util.List;

public class UserRepository {
    
    public interface UserCallback {
        void onSuccess(List<UserPageVO> users);
        void onError(String errorMessage);
    }
    
    public void getUsers(UserCallback callback) {
        UserQuery query = new UserQuery();
        query.setPageNum(1);
        query.setPageSize(20);
        
        UserAPI.getPage(query, new com.pei.dehaze.sdk.ApiCallback<PageResult<UserPageVO>>() {
            @Override
            public void onSuccess(PageResult<UserPageVO> data) {
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