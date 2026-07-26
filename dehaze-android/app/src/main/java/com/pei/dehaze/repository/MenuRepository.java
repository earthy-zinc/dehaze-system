package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.MenuAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.menu.MenuForm;
import com.pei.dehaze.sdk.model.menu.MenuQuery;
import com.pei.dehaze.sdk.model.menu.MenuVO;

import java.util.Collections;
import java.util.List;

public class MenuRepository {

    public void getMenuList(String keywords, RepositoryCallback<List<MenuVO>> callback) {
        MenuQuery query = new MenuQuery();
        query.setKeywords(keywords);
        MenuAPI.getList(query, RepositoryAdapters.wrap(callback));
    }

    public void getMenuOptions(RepositoryCallback<List<Option>> callback) {
        MenuAPI.getOptions(RepositoryAdapters.wrap(callback));
    }

    public void getMenuForm(long id, RepositoryCallback<MenuForm> callback) {
        MenuAPI.getFormData(id, RepositoryAdapters.wrap(callback));
    }

    public void addMenu(MenuForm form, RepositoryCallback<Void> callback) {
        MenuAPI.add(form, RepositoryAdapters.wrap(callback));
    }

    public void updateMenu(long id, MenuForm form, RepositoryCallback<Void> callback) {
        MenuAPI.update(id, form, RepositoryAdapters.wrap(callback));
    }

    public void deleteMenu(long id, RepositoryCallback<Void> callback) {
        MenuAPI.deleteByIds(Collections.singletonList(id), RepositoryAdapters.wrap(callback));
    }
}
