package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.DeptAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.dept.DeptForm;
import com.pei.dehaze.sdk.model.dept.DeptQuery;
import com.pei.dehaze.sdk.model.dept.DeptVO;

import java.util.List;

public class DeptRepository {

    public void getDepts(DeptQuery query, RepositoryCallback<List<DeptVO>> callback) {
        DeptAPI.getList(query, RepositoryAdapters.wrap(callback));
    }

    public void getDeptOptions(RepositoryCallback<List<Option>> callback) {
        DeptAPI.getOptions(RepositoryAdapters.wrap(callback));
    }

    public void getDeptForm(int id, RepositoryCallback<DeptForm> callback) {
        DeptAPI.getFormData(id, RepositoryAdapters.wrap(callback));
    }

    public void addDept(DeptForm form, RepositoryCallback<Void> callback) {
        DeptAPI.add(form, RepositoryAdapters.wrap(callback));
    }

    public void updateDept(int id, DeptForm form, RepositoryCallback<Void> callback) {
        DeptAPI.update(id, form, RepositoryAdapters.wrap(callback));
    }

    public void deleteDepts(List<Long> ids, RepositoryCallback<Void> callback) {
        DeptAPI.deleteByIds(ids, RepositoryAdapters.wrap(callback));
    }
}
