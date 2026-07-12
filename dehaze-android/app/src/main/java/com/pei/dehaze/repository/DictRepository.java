package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.DictAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.dict.DictForm;
import com.pei.dehaze.sdk.model.dict.DictPageVO;
import com.pei.dehaze.sdk.model.dict.DictQuery;
import com.pei.dehaze.sdk.model.dict.DictTypeForm;
import com.pei.dehaze.sdk.model.dict.DictTypePageVO;
import com.pei.dehaze.sdk.model.dict.DictTypeQuery;
import com.pei.dehaze.sdk.network.ApiException;

public class DictRepository {

    public interface DictTypePageCallback {
        void onSuccess(PageResult<DictTypePageVO> page);
        void onError(String errorMessage);
    }

    public interface DictTypeFormCallback {
        void onSuccess(DictTypeForm form);
        void onError(String errorMessage);
    }

    public interface DictTypeActionCallback {
        void onSuccess();
        void onError(String errorMessage);
    }

    public interface DictPageCallback {
        void onSuccess(PageResult<DictPageVO> page);
        void onError(String errorMessage);
    }

    public interface DictFormCallback {
        void onSuccess(DictForm form);
        void onError(String errorMessage);
    }

    public interface DictActionCallback {
        void onSuccess();
        void onError(String errorMessage);
    }

    public void getDictTypePage(DictTypeQuery query, DictTypePageCallback callback) {
        DictAPI.getDictTypePage(query, new ApiCallback<PageResult<DictTypePageVO>>() {
            @Override
            public void onSuccess(PageResult<DictTypePageVO> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("Error " + code + ": " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }

    public void getDictTypeForm(int id, DictTypeFormCallback callback) {
        DictAPI.getDictTypeForm(id, new ApiCallback<DictTypeForm>() {
            @Override
            public void onSuccess(DictTypeForm data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("Error " + code + ": " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }

    public void addDictType(DictTypeForm form, DictTypeActionCallback callback) {
        DictAPI.addDictType(form, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess();
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("Error " + code + ": " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }

    public void updateDictType(int id, DictTypeForm form, DictTypeActionCallback callback) {
        DictAPI.updateDictType(id, form, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess();
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("Error " + code + ": " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }

    public void deleteDictType(int id, DictTypeActionCallback callback) {
        DictAPI.deleteDictTypes(String.valueOf(id), new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess();
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("Error " + code + ": " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }

    public void getDictPage(DictQuery query, DictPageCallback callback) {
        DictAPI.getDictPage(query, new ApiCallback<PageResult<DictPageVO>>() {
            @Override
            public void onSuccess(PageResult<DictPageVO> data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("Error " + code + ": " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }

    public void getDictForm(int id, DictFormCallback callback) {
        DictAPI.getDictFormData(id, new ApiCallback<DictForm>() {
            @Override
            public void onSuccess(DictForm data) {
                callback.onSuccess(data);
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("Error " + code + ": " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }

    public void addDict(DictForm form, DictActionCallback callback) {
        DictAPI.addDict(form, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess();
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("Error " + code + ": " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }

    public void updateDict(int id, DictForm form, DictActionCallback callback) {
        DictAPI.updateDict(id, form, new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess();
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("Error " + code + ": " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }

    public void deleteDict(int id, DictActionCallback callback) {
        DictAPI.deleteDictByIds(String.valueOf(id), new ApiCallback<Void>() {
            @Override
            public void onSuccess(Void data) {
                callback.onSuccess();
            }

            @Override
            public void onError(String code, String message) {
                callback.onError("Error " + code + ": " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                callback.onError("Network error: " + e.getMessage());
            }
        });
    }
}
