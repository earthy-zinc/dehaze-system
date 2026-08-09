package com.pei.dehaze.ui.system;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import com.pei.dehaze.ui.common.BaseActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.databinding.ActivityManageListBinding;
import com.pei.dehaze.sdk.model.recommendation.RecommendationRule;
import com.pei.dehaze.ui.system.viewmodel.RecommendManageViewModel;

import java.util.List;

/**
 * 推荐管理（sys:recommendation:*）
 */
public class RecommendManageActivity extends BaseActivity {

    private RecommendManageViewModel viewModel;
    private ActivityManageListBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityManageListBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        setupToolbar(binding.toolbar, "推荐管理");

        // 推荐管理没有搜索和状态筛选，隐藏对应控件
        binding.etKeywords.setVisibility(View.GONE);
        binding.spinnerStatus.setVisibility(View.GONE);
        binding.btnSearch.setVisibility(View.GONE);
        binding.btnReset.setVisibility(View.GONE);
        binding.btnAdd.setVisibility(View.GONE);
        binding.btnPrev.setVisibility(View.GONE);
        binding.btnNext.setVisibility(View.GONE);
        binding.tvPageInfo.setVisibility(View.GONE);

        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(new RecommendManageAdapter());

        binding.swipeRefresh.setOnRefreshListener(this::loadData);
    }

    private void initViewModel() {
        viewModel = new ViewModelProvider(this).get(RecommendManageViewModel.class);
    }

    private void setupObservers() {
        viewModel.getItemList().observe(this, items -> {
            binding.recyclerView.getAdapter().notifyDataSetChanged();
            binding.tvEmpty.setVisibility(items == null || items.isEmpty() ? View.VISIBLE : View.GONE);
        });

        viewModel.getLoading().observe(this, isLoading ->
                binding.swipeRefresh.setRefreshing(isLoading != null && isLoading));

        observeError(viewModel);
        observeOperationResult(viewModel, null);
    }

    private void loadData() {
        viewModel.loadData();
    }

    private class RecommendManageAdapter extends RecyclerView.Adapter<RecommendManageAdapter.ViewHolder> {
        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext())
                    .inflate(android.R.layout.simple_list_item_2, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            List<?> list = viewModel.getItemList().getValue();
            if (list == null || position >= list.size()) return;
            Object obj = list.get(position);
            if (obj instanceof RecommendationRule) {
                RecommendationRule item = (RecommendationRule) obj;
                holder.text1.setText(item.getRuleName() != null ? item.getRuleName() : "未命名规则");
                String scene = item.getSceneType() != null ? item.getSceneType() : "";
                String weight = "权重: " + (item.getWeight() != null ? item.getWeight() : 0);
                String enabled = Boolean.TRUE.equals(item.getEnabled()) ? "启用" : "禁用";
                holder.text2.setText(scene + "  " + weight + "  " + enabled);
            } else {
                holder.text1.setText(obj.toString());
                holder.text2.setText("");
            }
        }

        @Override
        public int getItemCount() {
            List<?> list = viewModel.getItemList().getValue();
            return list != null ? list.size() : 0;
        }

        class ViewHolder extends RecyclerView.ViewHolder {
            TextView text1, text2;
            ViewHolder(View itemView) {
                super(itemView);
                text1 = itemView.findViewById(android.R.id.text1);
                text2 = itemView.findViewById(android.R.id.text2);
            }
        }
    }
}
