package com.pei.dehaze.ui.personal;

import android.content.Context;
import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.PopupMenu;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityFilesBinding;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.utils.ToastUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 我的文件 — 卡片化文件列表（缩略图 + 文件名 + 大小 + 上传时间 + 删除操作）
 */
public class FilesActivity extends AppCompatActivity {

    private ActivityFilesBinding binding;
    private FilesViewModel viewModel;
    private FilesAdapter adapter;
    private int currentPage = 1;
    private static final int PAGE_SIZE = 20;
    private boolean isLoading = false;
    private boolean hasMore = true;

    /** 图片类型的扩展名，用 Glide 加载缩略图 */
    private static final Set<String> IMAGE_EXTENSIONS = new HashSet<>(Arrays.asList(
            "jpg", "jpeg", "png", "gif", "bmp", "webp", "svg"));

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityFilesBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle("我的文件");
        }

        viewModel = new ViewModelProvider(this).get(FilesViewModel.class);
        adapter = new FilesAdapter(this, fileInfo -> showDeleteDialog(fileInfo));
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(adapter);

        binding.recyclerView.addOnScrollListener(new RecyclerView.OnScrollListener() {
            @Override
            public void onScrolled(@NonNull RecyclerView recyclerView, int dx, int dy) {
                LinearLayoutManager lm = (LinearLayoutManager) recyclerView.getLayoutManager();
                if (lm != null && lm.findLastVisibleItemPosition() + 1 >= adapter.getItemCount()
                        && hasMore && !isLoading) {
                    loadMore();
                }
            }
        });

        binding.swipeRefresh.setOnRefreshListener(() -> {
            currentPage = 1;
            hasMore = true;
            loadData();
        });

        viewModel.getFiles().observe(this, files -> adapter.submitList(files));
        viewModel.getLoading().observe(this, loading -> {
            isLoading = loading != null && loading;
            binding.swipeRefresh.setRefreshing(isLoading);
        });
        viewModel.getError().observe(this, msg -> {
            if (msg != null && !msg.isEmpty()) ToastUtils.showShort(this, msg);
        });

        loadData();
    }

    private void loadData() {
        FileAPI.getFilePage(currentPage, PAGE_SIZE, null,
                RepositoryAdapters.wrap(viewModel.createLoadingCallback(data -> {
                    List<FileInfo> list = data.getList();
                    adapter.submitList(list != null ? list : new ArrayList<>());
                    hasMore = list != null && list.size() >= PAGE_SIZE;
                })));
    }

    private void loadMore() {
        currentPage++;
        FileAPI.getFilePage(currentPage, PAGE_SIZE, null,
                RepositoryAdapters.wrap(viewModel.createLoadingCallback(data -> {
                    List<FileInfo> list = data.getList();
                    if (list != null) adapter.addAll(list);
                    hasMore = list != null && list.size() >= PAGE_SIZE;
                })));
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

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            finish();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    // region ViewModel

    public static class FilesViewModel extends BaseViewModel {
        private final androidx.lifecycle.MutableLiveData<List<FileInfo>> files =
                new androidx.lifecycle.MutableLiveData<>();
        public androidx.lifecycle.LiveData<List<FileInfo>> getFiles() { return files; }

        public <T> RepositoryCallback<T> createLoadingCallback(OnSuccess<T> onSuccess) {
            return withLoading(onSuccess);
        }

        public void deleteFile(Long fileId) {
            FileAPI.delete(fileId, RepositoryAdapters.wrap(withLoading(v -> {
                operationResult.postValue("文件已删除");
            })));
        }
    }

    // endregion

    // region Adapter

    static class FilesAdapter extends RecyclerView.Adapter<FilesAdapter.VH> {
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

        void addAll(List<FileInfo> newItems) {
            if (newItems != null) {
                int start = items.size();
                items.addAll(newItems);
                notifyItemRangeInserted(start, newItems.size());
            }
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

            // 缩略图：图片类型用 Glide 加载，其他用类型图标
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

            // 更多操作菜单
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
