package com.pei.dehaze.ui.tools;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.AlgorithmSelectAPI;
import com.pei.dehaze.sdk.model.algorithm_select.AlgorithmSelectNodeVO;
import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.ArrayList;
import java.util.List;

public class ToolsViewModel extends BaseViewModel {

    private final MutableLiveData<List<QuickEntry>> quickEntries = new MutableLiveData<>();
    private final MutableLiveData<List<FeatureItem>> featureItems = new MutableLiveData<>();
    private final MutableLiveData<String> searchKeyword = new MutableLiveData<>("");
    private final MutableLiveData<List<AlgorithmSelectNodeVO>> searchResults = new MutableLiveData<>();

    public ToolsViewModel() {
        initQuickEntries();
        initFeatureItems();
    }

    private void initQuickEntries() {
        List<QuickEntry> entries = new ArrayList<>();
        entries.add(new QuickEntry(1, "处理历史", "ic_image"));
        entries.add(new QuickEntry(2, "我的收藏", "ic_dataset"));
        entries.add(new QuickEntry(3, "批量处理", "ic_tools"));
        entries.add(new QuickEntry(4, "算法选择", "ic_algorithm"));
        quickEntries.setValue(entries);
    }

    private void initFeatureItems() {
        List<FeatureItem> items = new ArrayList<>();
        items.add(new FeatureItem(1, "图像输入", "ic_image", "upload"));
        items.add(new FeatureItem(2, "算法库", "ic_algorithm", "algorithm_list"));
        items.add(new FeatureItem(3, "数据集", "ic_dataset", "dataset"));
        items.add(new FeatureItem(4, "批量处理", "ic_tools", "batch"));
        items.add(new FeatureItem(5, "指标管理", "ic_dashboard", "metrics"));
        items.add(new FeatureItem(6, "API文档", "ic_captcha", "api_doc"));
        featureItems.setValue(items);
    }

    /**
     * 全局搜索：对接 AlgorithmSelectAPI.search，搜索算法（关键词/拼音/标签）
     */
    public void search(String keyword) {
        if (keyword == null || keyword.trim().isEmpty()) return;
        searchKeyword.setValue(keyword);
        loading.setValue(true);
        AlgorithmSelectAPI.search(keyword,
                RepositoryAdapters.wrap(new RepositoryCallback<List<AlgorithmSelectNodeVO>>() {
                    @Override
                    public void onSuccess(List<AlgorithmSelectNodeVO> data) {
                        loading.setValue(false);
                        searchResults.postValue(data != null ? data : new ArrayList<>());
                    }

                    @Override
                    public void onError(String errorMessage) {
                        error.postValue(errorMessage);
                        loading.setValue(false);
                    }
                }));
    }

    public void setSearchKeyword(String keyword) {
        searchKeyword.setValue(keyword);
    }

    public LiveData<List<QuickEntry>> getQuickEntries() {
        return quickEntries;
    }

    public LiveData<List<FeatureItem>> getFeatureItems() {
        return featureItems;
    }

    public LiveData<String> getSearchKeyword() {
        return searchKeyword;
    }

    public LiveData<List<AlgorithmSelectNodeVO>> getSearchResults() {
        return searchResults;
    }

    public static class QuickEntry {
        private final int id;
        private final String name;
        private final String iconName;

        public QuickEntry(int id, String name, String iconName) {
            this.id = id;
            this.name = name;
            this.iconName = iconName;
        }

        public int getId() { return id; }
        public String getName() { return name; }
        public String getIconName() { return iconName; }
    }

    public static class FeatureItem {
        private final int id;
        private final String name;
        private final String iconName;
        private final String action;

        public FeatureItem(int id, String name, String iconName, String action) {
            this.id = id;
            this.name = name;
            this.iconName = iconName;
            this.action = action;
        }

        public int getId() { return id; }
        public String getName() { return name; }
        public String getIconName() { return iconName; }
        public String getAction() { return action; }
    }
}
