package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.MemberAPI;
import com.pei.dehaze.sdk.model.member.LevelAdjustForm;
import com.pei.dehaze.sdk.model.member.MemberPageVO;
import com.pei.dehaze.sdk.model.member.MemberQuery;
import com.pei.dehaze.sdk.model.member.MemberStatusForm;

/**
 * 会员管理 ViewModel — 含列表查询、等级调整、状态切换、删除
 */
public class MemberManageViewModel extends BaseManageViewModel<MemberPageVO> {

    @Override
    public void loadData() {
        MemberQuery query = new MemberQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setKeywords(keywords.isEmpty() ? null : keywords);
        MemberAPI.getPage(query, RepositoryAdapters.wrap(withLoading(data -> {
            itemList.postValue(data.getList());
            total.postValue(data.getTotal());
        })));
    }

    public void adjustLevel(long userId, String levelCode, String reason) {
        LevelAdjustForm form = new LevelAdjustForm();
        form.setLevelCode(levelCode);
        form.setReason(reason);
        MemberAPI.adjustLevel(userId, form, RepositoryAdapters.wrap(withLoading(
                data -> {
                    operationResult.postValue("等级调整成功");
                    loadData();
                },
                errorMsg -> {
                    error.postValue(errorMsg);
                    loading.postValue(false);
                }
        )));
    }

    public void updateStatus(long userId, int newStatus) {
        MemberStatusForm form = new MemberStatusForm();
        form.setStatus(newStatus);
        MemberAPI.updateStatus(userId, form, RepositoryAdapters.wrap(withLoading(
                data -> {
                    operationResult.postValue("状态更新成功");
                    loadData();
                },
                errorMsg -> {
                    error.postValue(errorMsg);
                    loading.postValue(false);
                }
        )));
    }

    public void deleteMember(long userId) {
        // SDK MemberAPI 无直接 delete，通过 updateStatus 禁用作为软删除替代
        MemberStatusForm form = new MemberStatusForm();
        form.setStatus(0);
        MemberAPI.updateStatus(userId, form, RepositoryAdapters.wrap(withLoading(
                data -> {
                    operationResult.postValue("会员已禁用");
                    loadData();
                },
                errorMsg -> {
                    error.postValue(errorMsg);
                    loading.postValue(false);
                }
        )));
    }
}
