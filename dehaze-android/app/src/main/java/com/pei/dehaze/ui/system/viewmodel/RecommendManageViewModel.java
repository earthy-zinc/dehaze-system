package com.pei.dehaze.ui.system.viewmodel;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.RecommendationAPI;
import com.pei.dehaze.sdk.model.recommendation.RecommendationRule;

import java.util.List;

/**
 * 推荐管理 ViewModel
 */
public class RecommendManageViewModel extends BaseManageViewModel<RecommendationRule> {

    @Override
    public void loadData() {
        RecommendationAPI.getRules(RepositoryAdapters.wrap(withLoading(rules -> {
            itemList.postValue(rules);
            total.postValue((long) rules.size());
        })));
    }
}
