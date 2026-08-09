package com.pei.dehaze.ui.personal;

import android.content.Context;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.PopupMenu;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityFilesBinding;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.ui.common.BaseActivity;
import com.pei.dehaze.ui.common.BaseLoadMoreViewModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 我的文件 — 卡片化文件列表（缩略图 + 文件名 + 大小 + 上传时间 + 删除操作）
 *
 * <p>分页状态与请求由 {@link FilesViewModel} 持有，Activity 仅负责展示与交互。
 */
public class FilesActivity extends BaseActivity {

    private ActivityFilesBinding binding;
    private FilesViewModel viewModel;
    private FilesAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityFilesBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setupActionBar("我的文件");

        viewModel = new ViewModelProvider(this).get(FilesViewModel.class);
        adapter = new FilesAdapter(this, fileInfo -> showDeleteDialog(fileInfo));
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

        viewModel.getFiles().observe(this, list ->
                adapter.submitList(list));
        viewModel.getLoading().observe(this, loading ->
                binding.swipeRefresh.setRefreshing(Boolean.TRUE.equals(loading)));

        observeError(viewModel);
        observeOperationResult(viewModel, () -> viewModel.reload());

        viewModel.reload();
    }

    private void showDeleteDialog(FileInfo fileInfo) {
        new AlertDialog.Builder(this)
                .setTitle("删除文件")
                .setMessage("确定删除「" + (fileInfo.getName() != null ? fileInfo.getName() : "该文件") + "」吗？此操作不可撤销。")
                .setPositiveButton("删除", (dialog, which) -> {
                    if (fileInfo.getId() != null) {
                        viewModel.deleteFile(fileInfo.getId());
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    // region ViewModel

    public static class FilesViewModel extends BaseLoadMoreViewModel<FileInfo> {

        public FilesViewModel() {
            super(20);
        }

        @Override
        protected void loadPage() {
            FileAPI.getFilePage(pageNum, pageSize, null,
                    RepositoryAdapters.wrap(withLoading(data ->
                            onPageLoaded(data.getList(), data.getTotal()))));
        }

        public LiveData<List<FileInfo>> getFiles() {
            return itemList;
        }

        public void deleteFile(Long fileId) {
            FileAPI.delete(fileId, RepositoryAdapters.wrap(withLoading(v ->
                    operationResult.postValue("文件已删除"))));
        }
    }

    // endregion

    // region Adapter

    static class FilesAdapter extends RecyclerView.Adapter<FilesAdapter.VH> {
        private static final Set<String> IMAGE_EXTENSIONS = new HashSet<>(Arrays.asList(
                "jpg", "jpeg", "png", "gif", "bmp", "webp", "svg"));

        private final List<FileInfo> items = new ArrayList<>();
        private final Context context;
        private final OnFileActionListener actionListener;

        interface OnFileActionListener {
            void onDelete(FileInfo fileInfo);
        }

        FilesAdapter(Context context, OnFileActionListener actionListener) {
            this.context = context;
            this.actionListener = actionListener;
        }

        void submitList(List<FileInfo> newItems) {
            items.clear();
            if (newItems != null) items.addAll(newItems);
            notifyDataSetChanged();
        }

        @NonNull @Override
        public VH onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View v = android.view.LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_personal_file, parent, false);
            return new VH(v);
        }

        @Override
        public void onBindViewHolder(@NonNull VH holder, int position) {
            FileInfo item = items.get(position);
            holder.tvFileName.setText(item.getName() != null ? item.getName() : "未知文件");
            holder.tvFileSize.setText(formatSize(item.getSize()));
            holder.tvFileTime.setText(item.getCreateTime() != null ? item.getCreateTime() : "");

            if (isImageFile(item.getName())) {
                Glide.with(context)
                        .load(item.getUrl())
                        .placeholder(R.drawable.ic_file_placeholder)
                        .error(R.drawable.ic_file_placeholder)
                        .centerCrop()
                        .into(holder.ivThumbnail);
            } else {
                Glide.with(context).clear(holder.ivThumbnail);
                holder.ivThumbnail.setImageResource(R.drawable.ic_file_placeholder);
            }

            holder.ivMore.setOnClickListener(v -> {
                PopupMenu popup = new PopupMenu(v.getContext(), holder.ivMore);
                popup.getMenuInflater().inflate(R.menu.menu_file_item, popup.getMenu());
                popup.setOnMenuItemClickListener(menuItem -> {
                    if (menuItem.getItemId() == R.id.action_delete) {
                        if (actionListener != null) actionListener.onDelete(item);
                        return true;
                    }
                    return false;
                });
                popup.show();
            });
        }

        @Override
        public int getItemCount() { return items.size(); }

        private boolean isImageFile(String name) {
            if (name == null) return false;
            int dot = name.lastIndexOf('.');
            if (dot < 0) return false;
            return IMAGE_EXTENSIONS.contains(name.substring(dot + 1).toLowerCase());
        }

        static String formatSize(String sizeStr) {
            if (sizeStr == null || sizeStr.isEmpty()) return "";
            try {
                long bytes = Long.parseLong(sizeStr);
                if (bytes < 1024) return bytes + " B";
                if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
                return String.format("%.1f MB", bytes / (1024.0 * 1024));
            } catch (NumberFormatException e) {
                return sizeStr;
            }
        }

        static class VH extends RecyclerView.ViewHolder {
            ImageView ivThumbnail, ivMore;
            TextView tvFileName, tvFileSize, tvFileTime;

            VH(View v) {
                super(v);
                ivThumbnail = v.findViewById(R.id.iv_thumbnail);
                ivMore = v.findViewById(R.id.iv_more);
                tvFileName = v.findViewById(R.id.tv_file_name);
                tvFileSize = v.findViewById(R.id.tv_file_size);
                tvFileTime = v.findViewById(R.id.tv_file_time);
            }
        }
    }

    // endregion
}
