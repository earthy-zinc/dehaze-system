package com.pei.dehaze.ui.tools;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.pei.dehaze.ui.common.BaseViewModel;

import java.util.ArrayList;
import java.util.List;

public class ToolsViewModel extends BaseViewModel {

    private final MutableLiveData<List<QuickEntry>> quickEntries = new MutableLiveData<>();
    private final MutableLiveData<List<FeatureItem>> featureItems = new MutableLiveData<>();

    public ToolsViewModel() {
        initQuickEntries();
        initFeatureItems();
    }

    private void initQuickEntries() {
        List<QuickEntry> entries = new ArrayList<>();
        entries.add(new QuickEntry(1, "处理历史", "ic_history"));
        entries.add(new QuickEntry(2, "我的收藏", "ic_favorite"));
        entries.add(new QuickEntry(3, "批量处理", "ic_tools"));
        entries.add(new QuickEntry(4, "积分余额", "ic_credits"));
        quickEntries.setValue(entries);
    }

    private void initFeatureItems() {
        List<FeatureItem> items = new ArrayList<>();
        items.add(new FeatureItem(1, "图像输入", "ic_image", "upload"));
        items.add(new FeatureItem(2, "去雾处理", "ic_dehaze", "dehaze"));
        items.add(new FeatureItem(3, "批量处理", "ic_tools", "batch"));
        items.add(new FeatureItem(4, "数据集", "ic_dataset", "dataset"));
        items.add(new FeatureItem(5, "效果评估", "ic_metrics", "evaluation"));
        items.add(new FeatureItem(6, "指标管理", "ic_dashboard", "metrics"));
        featureItems.setValue(items);
    }

    public LiveData<List<QuickEntry>> getQuickEntries() {
        return quickEntries;
    }

    public LiveData<List<FeatureItem>> getFeatureItems() {
        return featureItems;
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
