package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.repository.MenuRepository;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.menu.MenuForm;
import com.pei.dehaze.sdk.model.menu.MenuVO;

import java.util.List;

public class MenuViewModel extends ViewModel {

    private final MenuRepository menuRepository;

    private final MutableLiveData<List<MenuVO>> menuList = new MutableLiveData<>();
    private final MutableLiveData<List<Option>> menuOptions = new MutableLiveData<>();
    private final MutableLiveData<MenuForm> menuForm = new MutableLiveData<>();
    private final MutableLiveData<Boolean> loading = new MutableLiveData<>();
    private final MutableLiveData<String> error = new MutableLiveData<>();
    private final MutableLiveData<String> actionResult = new MutableLiveData<>();

    public MenuViewModel() {
        menuRepository = new MenuRepository();
    }

    public void loadMenus(String keywords) {
        loading.setValue(true);
        menuRepository.getMenuList(keywords, new MenuRepository.MenuListCallback() {
            @Override
            public void onSuccess(List<MenuVO> menus) {
                menuList.postValue(menus);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void loadMenuOptions() {
        menuRepository.getMenuOptions(new MenuRepository.MenuOptionsCallback() {
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
        loading.setValue(true);
        menuRepository.getMenuForm(id, new MenuRepository.MenuFormCallback() {
            @Override
            public void onSuccess(MenuForm form) {
                menuForm.postValue(form);
                loading.postValue(false);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void addMenu(MenuForm form) {
        loading.setValue(true);
        menuRepository.addMenu(form, new MenuRepository.MenuActionCallback() {
            @Override
            public void onSuccess() {
                actionResult.postValue("新增菜单成功");
                loading.postValue(false);
                loadMenus(null);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void updateMenu(long id, MenuForm form) {
        loading.setValue(true);
        menuRepository.updateMenu(id, form, new MenuRepository.MenuActionCallback() {
            @Override
            public void onSuccess() {
                actionResult.postValue("修改菜单成功");
                loading.postValue(false);
                loadMenus(null);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
    }

    public void deleteMenu(long id) {
        loading.setValue(true);
        menuRepository.deleteMenu(id, new MenuRepository.MenuActionCallback() {
            @Override
            public void onSuccess() {
                actionResult.postValue("删除菜单成功");
                loading.postValue(false);
                loadMenus(null);
            }

            @Override
            public void onError(String errorMessage) {
                error.postValue(errorMessage);
                loading.postValue(false);
            }
        });
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

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }

    public LiveData<String> getActionResult() {
        return actionResult;
    }
}
