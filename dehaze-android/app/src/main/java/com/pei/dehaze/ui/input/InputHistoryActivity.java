package com.pei.dehaze.ui.input;

import androidx.appcompat.app.AlertDialog;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.bumptech.glide.Glide;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityInputHistoryBinding;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.input_history.InputHistoryForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryUpdateForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryVO;
import com.pei.dehaze.sdk.model.input_history.InputSource;
import com.pei.dehaze.sdk.model.input_history.SyncResultVO;
import com.pei.dehaze.ui.input.adapter.InputHistoryAdapter;
import com.pei.dehaze.ui.input.viewmodel.InputHistoryViewModel;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;
import com.pei.dehaze.utils.UriUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * 图像输入历史 Activity（列表 + 查询 + 新增 + 编辑 + 删除 + 批量删除 + 清空 + 同步 + 图片预览）
 */
public class InputHistoryActivity extends AppCompatActivity {

    private static final String[] SOURCE_LABELS = {"全部", "上传", "相机", "样本"};
    private static final InputSource[] SOURCE_VALUES = {null, InputSource.UPLOAD, InputSource.CAMERA, InputSource.SAMPLE};

    private InputHistoryViewModel viewModel;
    private InputHistoryAdapter adapter;
    private ActivityInputHistoryBinding binding;

    private InputHistoryFormDialog formDialog;
    private ActivityResultLauncher<String> imagePickerLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityInputHistoryBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                this::onImagePicked);

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        setSupportActionBar(binding.toolbar);
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        ArrayAdapter<String> sourceAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, SOURCE_LABELS);
        sourceAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        binding.spinnerSource.setAdapter(sourceAdapter);

        adapter = new InputHistoryAdapter();
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(adapter);

        binding.swipeRefresh.setOnRefreshListener(this::loadData);

        binding.btnSearch.setOnClickListener(v -> {
            String keywords = binding.etKeywords.getText().toString().trim();
            InputSource source = SOURCE_VALUES[binding.spinnerSource.getSelectedItemPosition()];
            boolean favoriteOnly = binding.cbFavoriteOnly.isChecked();
            viewModel.setQueryParams(keywords, source, favoriteOnly);
            loadData();
        });

        binding.btnReset.setOnClickListener(v -> {
            binding.etKeywords.setText("");
            binding.spinnerSource.setSelection(0);
            binding.cbFavoriteOnly.setChecked(false);
            viewModel.resetQuery();
            loadData();
        });

        binding.btnAdd.setOnClickListener(v -> showFormDialog(null));

        binding.btnBatchDelete.setOnClickListener(v -> {
            if (!adapter.isSelectionMode()) {
                adapter.setSelectionMode(true);
                updateSelectionModeUI(true);
                ToastUtils.showShort(this, "长按或勾选要删除的记录");
            } else {
                confirmBatchDelete();
            }
        });

        binding.btnCancelSelect.setOnClickListener(v -> {
            adapter.clearSelection();
            adapter.setSelectionMode(false);
            updateSelectionModeUI(false);
            updatePageInfo();
        });

        binding.btnSelectAll.setOnClickListener(v -> adapter.selectAll());

        binding.btnClear.setOnClickListener(v -> confirmClear());

        binding.btnSync.setOnClickListener(v -> confirmSync());

        adapter.setActionListener(new InputHistoryAdapter.OnHistoryActionListener() {
            @Override
            public void onItemClick(InputHistoryVO item) {
                showImagePreviewDialog(item);
            }

            @Override
            public void onEdit(InputHistoryVO item) {
                showFormDialog(item);
            }

            @Override
            public void onDelete(InputHistoryVO item) {
                confirmDelete(item);
            }

            @Override
            public void onToggleFavorite(InputHistoryVO item) {
                InputHistoryUpdateForm form = new InputHistoryUpdateForm();
                form.setIsFavorite(item.getIsFavorite() != null && item.getIsFavorite() == 1 ? 0 : 1);
                viewModel.updateHistory(item.getId(), form);
            }
        });

        adapter.setSelectionListener(selectedIds ->
                binding.btnBatchDelete.setText(selectedIds.isEmpty() ? "批量删除" : "删除选中(" + selectedIds.size() + ")"));
    }

    private void updateSelectionModeUI(boolean selectionMode) {
        binding.btnCancelSelect.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        binding.btnSelectAll.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        binding.btnAdd.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        binding.btnClear.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        binding.btnSync.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        binding.btnBatchDelete.setText(selectionMode ? "删除选中" : "批量删除");
    }

    private void initViewModel() {
        viewModel = new ViewModelProvider(this).get(InputHistoryViewModel.class);
    }

    private void setupObservers() {
        viewModel.getHistoryList().observe(this, items -> {
            adapter.submitList(items);
            updatePageInfo();
            binding.tvEmpty.setVisibility(items == null || items.isEmpty() ? View.VISIBLE : View.GONE);
        });

        viewModel.getTotal().observe(this, total -> updatePageInfo());

        viewModel.getLoading().observe(this, isLoading ->
                binding.swipeRefresh.setRefreshing(isLoading != null && isLoading));

        viewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(this, errorMessage);
                viewModel.clearError();
            }
        });

        viewModel.getOperationResult().observe(this, result -> {
            if (result != null && !result.isEmpty()) {
                ToastUtils.showShort(this, result);
                viewModel.clearOperationResult();
                if (result.startsWith("删除") && adapter.isSelectionMode()) {
                    adapter.clearSelection();
                    adapter.setSelectionMode(false);
                    updateSelectionModeUI(false);
                }
            }
        });

        viewModel.getSyncResult().observe(this, result -> {
            if (result != null) {
                showSyncResultDialog(result);
                viewModel.clearSyncResult();
            }
        });

        viewModel.getUploadedFile().observe(this, fileInfo -> {
            if (fileInfo != null) {
                if (formDialog != null) {
                    formDialog.onFileUploaded(fileInfo);
                }
                viewModel.clearUploadedFile();
            }
        });
    }

    /**
     * 图片选择回调
     */
    private void onImagePicked(Uri uri) {
        if (uri == null) {
            ToastUtils.showShort(this, "未选择图片");
            return;
        }
        File file = UriUtils.copyToCache(this, uri);
        if (file == null) {
            ToastUtils.showShort(this, "无法读取图片文件");
            return;
        }
        ToastUtils.showShort(this, "开始上传图片...");
        viewModel.uploadFile(file);
    }

    private void loadData() {
        viewModel.loadHistory();
    }

    private void updatePageInfo() {
        if (adapter != null && adapter.isSelectionMode()) {
            int count = adapter.getSelectedIds().size();
            binding.tvPageInfo.setText("已选中 " + count + " 项");
            return;
        }
        long totalVal = viewModel.getTotal().getValue() != null ? viewModel.getTotal().getValue() : 0L;
        int pageNum = viewModel.getPageNum();
        int pageSize = viewModel.getPageSize();
        int totalPages = Math.max(1, (int) Math.ceil(totalVal * 1.0 / pageSize));
        binding.tvPageInfo.setText("第 " + pageNum + " 页 / 共 " + totalPages + " 页 (共 " + totalVal + " 条)");
    }

    private void showFormDialog(InputHistoryVO existing) {
        formDialog = new InputHistoryFormDialog(this, new InputHistoryFormDialog.Callback() {
            @Override
            public void onPickImage() {
                imagePickerLauncher.launch("image/*");
            }

            @Override
            public void onCreate(InputHistoryForm form) {
                viewModel.createHistory(form);
            }

            @Override
            public void onUpdate(long historyId, InputHistoryUpdateForm form) {
                viewModel.updateHistory(historyId, form);
            }
        });
        formDialog.show(existing);
    }

    private void confirmDelete(InputHistoryVO item) {
        new AlertDialog.Builder(this)
                .setTitle("删除确认")
                .setMessage("确认删除该历史记录吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) -> viewModel.deleteHistory(item.getId()))
                .setNegativeButton("取消", null)
                .show();
    }

    private void confirmBatchDelete() {
        Set<Long> selectedIds = adapter.getSelectedIds();
        if (selectedIds.isEmpty()) {
            ToastUtils.showShort(this, "请先选择要删除的记录");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("批量删除确认")
                .setMessage("确认删除选中的 " + selectedIds.size() + " 条记录吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) ->
                        viewModel.batchDeleteHistory(new ArrayList<>(selectedIds)))
                .setNegativeButton("取消", null)
                .show();
    }

    private void confirmClear() {
        new AlertDialog.Builder(this)
                .setTitle("清空历史")
                .setMessage("确认清空所有历史记录吗？此操作不可恢复！")
                .setPositiveButton("确定", (dialog, which) -> viewModel.clearHistory())
                .setNegativeButton("取消", null)
                .show();
    }

    private void confirmSync() {
        List<InputHistoryVO> items = viewModel.getHistoryList().getValue();
        if (items == null || items.isEmpty()) {
            ToastUtils.showShort(this, "暂无可同步的历史记录");
            return;
        }
        List<InputHistoryForm> syncItems = new ArrayList<>();
        for (InputHistoryVO item : items) {
            if (item.getSyncStatus() == null || item.getSyncStatus() == 0) {
                InputHistoryForm form = new InputHistoryForm();
                form.setOriginalImageUrl(item.getOriginalImageUrl());
                form.setAlgorithmId(item.getAlgorithmId());
                form.setAlgorithmName(item.getAlgorithmName());
                form.setAlgorithmParams(item.getAlgorithmParams());
                form.setProcessingTime(item.getProcessingTime());
                form.setStatus(item.getStatus());
                form.setInputSource(item.getInputSource());
                form.setIsFavorite(item.getIsFavorite());
                syncItems.add(form);
            }
        }
        if (syncItems.isEmpty()) {
            ToastUtils.showShort(this, "没有未同步的记录");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("同步到数据集")
                .setMessage("确认将 " + syncItems.size() + " 条未同步记录同步到数据集吗？")
                .setPositiveButton("确定", (dialog, which) -> viewModel.syncHistory(syncItems))
                .setNegativeButton("取消", null)
                .show();
    }

    private void showSyncResultDialog(SyncResultVO result) {
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_sync_result, null);
        TextView tvSummary = view.findViewById(R.id.tv_sync_summary);
        TextView tvMessage = view.findViewById(R.id.tv_sync_message);
        tvSummary.setText("同步完成：成功 " + result.getSynced() + " 条，失败 " + result.getFailed() + " 条");
        tvMessage.setText(StringUtils.safe(result.getMessage()));

        new AlertDialog.Builder(this)
                .setTitle("同步结果")
                .setView(view)
                .setPositiveButton("确定", (dialog, which) -> loadData())
                .show();
    }

    private void showImagePreviewDialog(InputHistoryVO item) {
        View view = LayoutInflater.from(this).inflate(R.layout.dialog_image_preview, null);
        ImageView ivPreview = view.findViewById(R.id.iv_preview);
        String url = item.getResultImageUrl();
        if (TextUtils.isEmpty(url)) {
            url = item.getOriginalImageUrl();
        }
        Glide.with(this)
                .load(DehazeSDK.getInstance().resolveUrl(url))
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(ivPreview);
        new AlertDialog.Builder(this)
                .setView(view)
                .setPositiveButton("关闭", null)
                .show();
    }
}
