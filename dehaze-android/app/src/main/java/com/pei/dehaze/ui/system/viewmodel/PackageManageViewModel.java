package com.pei.dehaze.ui.system.viewmodel;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.PackageAPI;
import com.pei.dehaze.sdk.model.pkg.PackageForm;
import com.pei.dehaze.sdk.model.pkg.PackagePageVO;
import com.pei.dehaze.sdk.model.pkg.PackageQuery;

/**
 * 套餐管理 ViewModel — 含列表查询、新增/编辑、上下架、删除
 */
public class PackageManageViewModel extends BaseManageViewModel<PackagePageVO> {

    @Override
    public void loadData() {
        PackageQuery query = new PackageQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setName(keywords.isEmpty() ? null : keywords);
        PackageAPI.getPage(query, RepositoryAdapters.wrap(withLoading(data -> {
            itemList.postValue(data.getList());
            total.postValue(data.getTotal());
        })));
    }

    public void addPackage(PackageForm form) {
        PackageAPI.add(form, RepositoryAdapters.wrap(withLoading(
                data -> {
                    operationResult.postValue("套餐新增成功");
                    loadData();
                },
                errorMsg -> {
                    error.postValue(errorMsg);
                    loading.postValue(false);
                }
        )));
    }

    public void updatePackage(long id, PackageForm form) {
        PackageAPI.update(id, form, RepositoryAdapters.wrap(withLoading(
                data -> {
                    operationResult.postValue("套餐更新成功");
                    loadData();
                },
                errorMsg -> {
                    error.postValue(errorMsg);
                    loading.postValue(false);
                }
        )));
    }

    public void toggleStatus(long id, int newStatus) {
        PackageAPI.updateStatus(id, newStatus, RepositoryAdapters.wrap(withLoading(
                data -> {
                    operationResult.postValue(newStatus == 1 ? "套餐已上架" : "套餐已下架");
                    loadData();
                },
                errorMsg -> {
                    error.postValue(errorMsg);
                    loading.postValue(false);
                }
        )));
    }

    public void deletePackage(long id) {
        PackageAPI.deleteByIds(String.valueOf(id), RepositoryAdapters.wrap(withLoading(
                data -> {
                    operationResult.postValue("套餐已删除");
                    loadData();
                },
                errorMsg -> {
                    error.postValue(errorMsg);
                    loading.postValue(false);
                }
        )));
    }
}
