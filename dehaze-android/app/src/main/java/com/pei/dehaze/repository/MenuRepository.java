package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.MenuAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.menu.MenuForm;
import com.pei.dehaze.sdk.model.menu.MenuQuery;
import com.pei.dehaze.sdk.model.menu.MenuVO;
import com.pei.dehaze.sdk.network.ApiException;

import java.util.List;

public class MenuRepository {

    public interface MenuListCallback {
        void onSuccess(List<MenuVO> menus);
        void onError(String errorMessage);
    }

    public interface MenuOptionsCallback {
        void onSuccess(List<Option> options);
        void onError(String errorMessage);
    }

    public interface MenuFormCallback {
        void onSuccess(MenuForm form);
        void onError(String errorMessage);
    }

    public interface MenuActionCallback {
        void onSuccess();
        void onError(String errorMessage);
    }

    public void getMenuList(String keywords, MenuListCallback callback) {
        MenuQuery query = new MenuQuery();
        query.setKeywords(keywords);
        MenuAPI.getList(query, new ApiCallback<List<MenuVO>>() {
            @Override
            public void onSuccess(List<MenuVO> data) {
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

    public void getMenuOptions(MenuOptionsCallback callback) {
        MenuAPI.getOptions(new ApiCallback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> data) {
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

    public void getMenuForm(long id, MenuFormCallback callback) {
        MenuAPI.getFormData(id, new ApiCallback<MenuForm>() {
            @Override
            public void onSuccess(MenuForm data) {
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

    public void addMenu(MenuForm form, MenuActionCallback callback) {
        MenuAPI.add(form, new ApiCallback<Void>() {
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

    public void updateMenu(long id, MenuForm form, MenuActionCallback callback) {
        MenuAPI.update(id, form, new ApiCallback<Void>() {
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

    public void deleteMenu(long id, MenuActionCallback callback) {
        MenuAPI.deleteById(id, new ApiCallback<Void>() {
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
