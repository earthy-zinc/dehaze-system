package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.MenuRepository;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.menu.MenuForm;
import com.pei.dehaze.sdk.model.menu.MenuVO;

import java.util.List;

public class MenuViewModel extends BaseViewModel {

    private final MenuRepository menuRepository;

    private final MutableLiveData<List<MenuVO>> menuList = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> menuOptions = new MutableLiveData<>();
    private final MutableLiveData<MenuForm> menuForm = new MutableLiveData<>();

    public MenuViewModel() {
        menuRepository = new MenuRepository();
    }

    public void loadMenus(String keywords) {
        menuRepository.getMenuList(keywords, withLoading(menuList::postValue));
    }

    public void loadMenuOptions() {
        menuRepository.getMenuOptions(new RepositoryCallback<List<Option>>() {
            @Override
            public void onSuccess(List<Option> options) {
                menuOptions.postValue(options);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
            }
        });
    }

    public void loadMenuForm(long id) {
        menuRepository.getMenuForm(id, withLoading(menuForm::postValue));
    }

    public void addMenu(MenuForm form) {
        menuRepository.addMenu(form, withLoading(v -> {
            operationResult.postValue("新增菜单成功");
            loadMenus(null);
        }));
    }

    public void updateMenu(long id, MenuForm form) {
        menuRepository.updateMenu(id, form, withLoading(v -> {
            operationResult.postValue("修改菜单成功");
            loadMenus(null);
        }));
    }

    public void deleteMenu(long id) {
        menuRepository.deleteMenu(id, withLoading(v -> {
            operationResult.postValue("删除菜单成功");
            loadMenus(null);
        }));
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
