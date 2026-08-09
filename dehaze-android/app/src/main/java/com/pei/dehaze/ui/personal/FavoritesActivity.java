package com.pei.dehaze.ui.personal;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivitySimpleListBinding;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.FavoriteAPI;
import com.pei.dehaze.sdk.model.favorite.FavoriteQuery;
import com.pei.dehaze.sdk.model.favorite.FavoriteVO;
import com.pei.dehaze.ui.common.BaseActivity;
import com.pei.dehaze.ui.common.BaseLoadMoreViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 我的收藏 — 卡片化列表（缩略图 + 标题 + 类型标签 + 收藏时间 + 取消收藏）
 *
 * <p>分页状态与请求由 {@link FavoriteViewModel} 持有，Activity 仅负责展示与交互。
 */
public class FavoritesActivity extends BaseActivity {

    private ActivitySimpleListBinding binding;
    private FavoriteViewModel viewModel;
    private FavoriteAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivitySimpleListBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setupActionBar("我的收藏");

        viewModel = new ViewModelProvider(this).get(FavoriteViewModel.class);
        adapter = new FavoriteAdapter(item -> showUnfavoriteDialog(item));
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(adapter);

        binding.recyclerView.addOnScrollListener(new RecyclerView.OnScrollListener() {
            @Override
            public void onScrolled(@NonNull RecyclerView recyclerView, int dx, int dy) {
                LinearLayoutManager lm = (LinearLayoutManager) recyclerView.getLayoutManager();
                if (lm != null && lm.findLastVisibleItemPosition() + 1 >= adapter.getItemCount()
                        && !Boolean.TRUE.equals(viewModel.getLoading().getValue())) {
                    viewModel.loadMore();
                }
            }
        });

        binding.swipeRefresh.setOnRefreshListener(() -> viewModel.reload());

        viewModel.getFavorites().observe(this, list -> {
            adapter.submitList(list);
            if (list == null || list.isEmpty()) {
                binding.emptyView.setVisibility(View.VISIBLE);
                binding.emptyText.setText("暂无收藏");
            } else {
                binding.emptyView.setVisibility(View.GONE);
            }
        });
        viewModel.getLoading().observe(this, loading ->
                binding.swipeRefresh.setRefreshing(Boolean.TRUE.equals(loading)));

        observeError(viewModel);
        observeOperationResult(viewModel, () -> viewModel.reload());

        viewModel.reload();
    }

    private void showUnfavoriteDialog(FavoriteVO item) {
        new AlertDialog.Builder(this)
                .setTitle("取消收藏")
                .setMessage("确定取消收藏「" + (item.getTargetName() != null ? item.getTargetName() : "该项") + "」吗？")
                .setPositiveButton("确定", (dialog, which) -> {
                    if (item.getId() != null) {
                        viewModel.removeFavorite(item.getId());
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    // region ViewModel

    public static class FavoriteViewModel extends BaseLoadMoreViewModel<FavoriteVO> {

        public FavoriteViewModel() {
            super(20);
        }

        @Override
        protected void loadPage() {
            FavoriteQuery query = new FavoriteQuery();
            query.setPageNum(pageNum);
            query.setPageSize(pageSize);
            FavoriteAPI.getPage(query, RepositoryAdapters.wrap(withLoading(data ->
                    onPageLoaded(data.getList(), data.getTotal()))));
        }

        public LiveData<List<FavoriteVO>> getFavorites() {
            return itemList;
        }

        public void removeFavorite(Long id) {
            FavoriteAPI.deleteByIds(Collections.singletonList(id),
                    RepositoryAdapters.wrap(withLoading(v ->
                            operationResult.postValue("已取消收藏"))));
        }
    }

    // endregion

    // region Adapter

    static class FavoriteAdapter extends RecyclerView.Adapter<FavoriteAdapter.VH> {
        private final List<FavoriteVO> items = new ArrayList<>();
        private final OnFavoriteActionListener actionListener;

        interface OnFavoriteActionListener {
            void onUnfavorite(FavoriteVO item);
        }

        FavoriteAdapter(OnFavoriteActionListener actionListener) {
            this.actionListener = actionListener;
        }

        void submitList(List<FavoriteVO> newItems) {
            items.clear();
            if (newItems != null) items.addAll(newItems);
            notifyDataSetChanged();
        }

        @NonNull @Override
        public VH onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View v = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_personal_favorite, parent, false);
            return new VH(v);
        }

        @Override
        public void onBindViewHolder(@NonNull VH holder, int position) {
            FavoriteVO item = items.get(position);
            holder.tvName.setText(item.getTargetName() != null ? item.getTargetName() : "未命名");
            holder.tvType.setText(item.getTargetType() != null ? item.getTargetType() : "--");
            holder.tvTime.setText(item.getCreateTime() != null ? item.getCreateTime() : "");

            // 缩略图
            if (item.getTargetThumbnail() != null && !item.getTargetThumbnail().isEmpty()) {
                Glide.with(holder.ivThumbnail.getContext())
                        .load(item.getTargetThumbnail())
                        .placeholder(R.drawable.ic_file_placeholder)
                        .error(R.drawable.ic_file_placeholder)
                        .centerCrop()
                        .into(holder.ivThumbnail);
            } else {
                holder.ivThumbnail.setImageResource(R.drawable.ic_file_placeholder);
            }

            // 取消收藏按钮
            holder.btnUnfavorite.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onUnfavorite(item);
            });
        }

        @Override
        public int getItemCount() { return items.size(); }

        static class VH extends RecyclerView.ViewHolder {
            ImageView ivThumbnail;
            TextView tvName, tvType, tvTime, btnUnfavorite;

            VH(View v) {
                super(v);
                ivThumbnail = v.findViewById(R.id.iv_thumbnail);
                tvName = v.findViewById(R.id.tv_favorite_name);
                tvType = v.findViewById(R.id.tv_favorite_type);
                tvTime = v.findViewById(R.id.tv_favorite_time);
                btnUnfavorite = v.findViewById(R.id.btn_unfavorite);
            }
        }
    }

    // endregion
}
