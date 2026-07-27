package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.MenuAPI;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.menu.MenuForm;
import com.pei.dehaze.sdk.model.menu.MenuQuery;
import com.pei.dehaze.sdk.model.menu.MenuVO;

import java.util.Collections;
import java.util.List;

public class MenuViewModel extends BaseViewModel {

    private final MutableLiveData<List<MenuVO>> menuList = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> menuOptions = new MutableLiveData<>();
    private final MutableLiveData<MenuForm> menuForm = new MutableLiveData<>();

    public void loadMenus(String keywords) {
        MenuQuery query = new MenuQuery();
        query.setKeywords(keywords);
        MenuAPI.getList(query, RepositoryAdapters.wrap(withLoading(menuList::postValue)));
    }

    public void loadMenuOptions() {
        MenuAPI.getOptions(RepositoryAdapters.wrap(new RepositoryCallback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> options) {
                menuOptions.postValue(options);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        }));
    }

    public void loadMenuForm(long id) {
        MenuAPI.getFormData(id, RepositoryAdapters.wrap(withLoading(menuForm::postValue)));
    }

    public void addMenu(MenuForm form) {
        MenuAPI.add(form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("新增菜单成功");
            loadMenus(null);
        })));
    }

    public void updateMenu(long id, MenuForm form) {
        MenuAPI.update(id, form, RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("修改菜单成功");
            loadMenus(null);
        })));
    }

    public void deleteMenu(long id) {
        MenuAPI.deleteByIds(Collections.singletonList(id), RepositoryAdapters.wrap(withLoading(v -> {
            operationResult.postValue("删除菜单成功");
            loadMenus(null);
        })));
    }

    public LiveData<List<MenuVO>> getMenuList() {
        return menuList;
    }

    public LiveData<List<Option>> getMenuOptions() {
        return menuOptions;
    }

    public LiveData<MenuForm> getMenuForm() {
        return menuForm;
    }
}
