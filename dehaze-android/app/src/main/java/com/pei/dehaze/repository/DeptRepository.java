package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.DeptAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.dept.DeptForm;
import com.pei.dehaze.sdk.model.dept.DeptQuery;
import com.pei.dehaze.sdk.model.dept.DeptVO;
import com.pei.dehaze.sdk.network.ApiException;

import java.util.List;

public class DeptRepository {

    public interface Callback<T> {
        void onSuccess(T data);
        void onError(String errorMessage);
    }

    public void getDepts(DeptQuery query, Callback<List<DeptVO>> callback) {
        DeptAPI.getList(query, new ApiCallback<List<DeptVO>>() {
            @Override
            public void onSuccess(List<DeptVO> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getDeptOptions(Callback<List<Option>> callback) {
        DeptAPI.getOptions(new ApiCallback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void getDeptForm(int id, Callback<DeptForm> callback) {
        DeptAPI.getFormData(id, new ApiCallback<DeptForm>() {
            @Override
            public void onSuccess(DeptForm data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void addDept(DeptForm form, Callback<Void> callback) {
        DeptAPI.add(form, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void updateDept(int id, DeptForm form, Callback<Void> callback) {
        DeptAPI.update(id, form, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }

    public void deleteDepts(String ids, Callback<Void> callback) {
        DeptAPI.deleteByIds(ids, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("[" + code + "] " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError(e.getMessage());
            }
        });
    }
}
