package com.pei.dehaze.ui.input;

import android.app.AlertDialog;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.bumptech.glide.Glide;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.checkbox.MaterialCheckBox;
import com.google.android.material.textfield.TextInputEditText;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.input_history.InputHistoryForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryUpdateForm;
import com.pei.dehaze.sdk.model.input_history.InputHistoryVO;
import com.pei.dehaze.sdk.model.input_history.SyncResultVO;
import com.pei.dehaze.ui.input.adapter.InputHistoryAdapter;
import com.pei.dehaze.ui.input.viewmodel.InputHistoryViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * 图像输入历史 Activity（列表 + 查询 + 新增 + 编辑 + 删除 + 批量删除 + 清空 + 同步 + 图片预览）
 */
public class InputHistoryActivity extends AppCompatActivity {

    private static final String[] STATUS_LABELS = {"成功", "失败", "处理中"};
    private static final Integer[] STATUS_VALUES = {1, 2, 3};
    private static final String[] SOURCE_LABELS = {"全部", "上传", "相机", "样本"};
    private static final String[] SOURCE_VALUES = {null, "upload", "camera", "sample"};
    private static final String[] FORM_SOURCE_LABELS = {"上传", "相机", "样本"};
    private static final String[] FORM_SOURCE_VALUES = {"upload", "camera", "sample"};

    private InputHistoryViewModel viewModel;
    private InputHistoryAdapter adapter;

    private Toolbar toolbar;
    private SwipeRefreshLayout swipeRefresh;
    private RecyclerView recyclerView;
    private TextView tvPageInfo;
    private Spinner spinnerSource;
    private MaterialCheckBox cbFavoriteOnly;
    private MaterialButton btnAdd;
    private MaterialButton btnBatchDelete;
    private MaterialButton btnCancelSelect;
    private MaterialButton btnSelectAll;
    private MaterialButton btnClear;
    private MaterialButton btnSync;

    /** 当前编辑的历史记录ID，0 表示新增模式 */
    private long editingHistoryId = 0;
    /** 当前表单中已上传的原始图片 URL */
    private String currentOriginalUrl;
    /** 当前表单中显示 URL 的 TextView 引用 */
    private TextView currentOriginalUrlView;

    private ActivityResultLauncher<String> imagePickerLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_input_history);

        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                this::onImagePicked);

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        toolbar = findViewById(R.id.toolbar);
        recyclerView = findViewById(R.id.recycler_view);
        swipeRefresh = findViewById(R.id.swipe_refresh);
        tvPageInfo = findViewById(R.id.tv_page_info);
        spinnerSource = findViewById(R.id.spinner_source);
        cbFavoriteOnly = findViewById(R.id.cb_favorite_only);
        btnAdd = findViewById(R.id.btn_add);
        btnBatchDelete = findViewById(R.id.btn_batch_delete);
        btnCancelSelect = findViewById(R.id.btn_cancel_select);
        btnSelectAll = findViewById(R.id.btn_select_all);
        btnClear = findViewById(R.id.btn_clear);
        btnSync = findViewById(R.id.btn_sync);

        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(v -> finish());

        ArrayAdapter<String> sourceAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, SOURCE_LABELS);
        sourceAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerSource.setAdapter(sourceAdapter);

        adapter = new InputHistoryAdapter();
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(adapter);

        swipeRefresh.setOnRefreshListener(this::loadData);

        findViewById(R.id.btn_search).setOnClickListener(v -> {
            String keywords = ((android.widget.EditText) findViewById(R.id.et_keywords))
                    .getText().toString().trim();
            String source = SOURCE_VALUES[spinnerSource.getSelectedItemPosition()];
            boolean favoriteOnly = cbFavoriteOnly.isChecked();
            viewModel.setQueryParams(keywords, source, favoriteOnly);
            loadData();
        });

        findViewById(R.id.btn_reset).setOnClickListener(v -> {
            ((android.widget.EditText) findViewById(R.id.et_keywords)).setText("");
            spinnerSource.setSelection(0);
            cbFavoriteOnly.setChecked(false);
            viewModel.resetQuery();
            loadData();
        });

        btnAdd.setOnClickListener(v -> showFormDialog(null));

        btnBatchDelete.setOnClickListener(v -> {
            if (!adapter.isSelectionMode()) {
                adapter.setSelectionMode(true);
                updateSelectionModeUI(true);
                ToastUtils.showShort(this, "长按或勾选要删除的记录");
            } else {
                confirmBatchDelete();
            }
        });

        btnCancelSelect.setOnClickListener(v -> {
            adapter.clearSelection();
            adapter.setSelectionMode(false);
            updateSelectionModeUI(false);
            updatePageInfo();
        });

        btnSelectAll.setOnClickListener(v -> adapter.selectAll());

        btnClear.setOnClickListener(v -> confirmClear());

        btnSync.setOnClickListener(v -> confirmSync());

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
                btnBatchDelete.setText(selectedIds.isEmpty() ? "批量删除" : "删除选中(" + selectedIds.size() + ")"));
    }

    private void updateSelectionModeUI(boolean selectionMode) {
        btnCancelSelect.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        btnSelectAll.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        btnAdd.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        btnClear.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        btnSync.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        btnBatchDelete.setText(selectionMode ? "删除选中" : "批量删除");
    }

    private void initViewModel() {
        viewModel = new ViewModelProvider(this).get(InputHistoryViewModel.class);
    }

    private void setupObservers() {
        viewModel.getHistoryList().observe(this, items -> {
            adapter.submitList(items);
            updatePageInfo();
        });

        viewModel.getTotal().observe(this, total -> updatePageInfo());

        viewModel.getLoading().observe(this, isLoading ->
                swipeRefresh.setRefreshing(isLoading != null && isLoading));

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
                onFileUploaded(fileInfo);
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
        File file = uriToFile(uri);
        if (file == null) {
            ToastUtils.showShort(this, "无法读取图片文件");
            return;
        }
        ToastUtils.showShort(this, "开始上传图片...");
        viewModel.uploadFile(file);
    }

    /**
     * 文件上传成功回调
     */
    private void onFileUploaded(FileInfo fileInfo) {
        currentOriginalUrl = fileInfo.getUrl();
        if (currentOriginalUrlView != null) {
            currentOriginalUrlView.setVisibility(View.VISIBLE);
            currentOriginalUrlView.setText("已上传: " + fileInfo.getUrl());
        }
        ToastUtils.showShort(this, "图片上传成功");

        // 编辑模式下，自动调用 updateHistory 更新原图片 URL
        if (editingHistoryId > 0) {
            InputHistoryUpdateForm form = new InputHistoryUpdateForm();
            form.setOriginalImageUrl(currentOriginalUrl);
            viewModel.updateHistory(editingHistoryId, form);
        }
    }

    private void loadData() {
        viewModel.loadHistory();
    }

    private void updatePageInfo() {
        if (adapter != null && adapter.isSelectionMode()) {
            int count = adapter.getSelectedIds().size();
            tvPageInfo.setText("已选中 " + count + " 项");
            return;
        }
        long totalVal = viewModel.getTotal().getValue() != null ? viewModel.getTotal().getValue() : 0L;
        int pageNum = viewModel.getPageNum();
        int pageSize = viewModel.getPageSize();
        int totalPages = Math.max(1, (int) Math.ceil(totalVal * 1.0 / pageSize));
        tvPageInfo.setText("第 " + pageNum + " 页 / 共 " + totalPages + " 页 (共 " + totalVal + " 条)");
    }

    private void showFormDialog(InputHistoryVO existing) {
        boolean isEdit = existing != null;
        editingHistoryId = isEdit ? existing.getId() : 0;
        currentOriginalUrl = isEdit ? existing.getOriginalImageUrl() : null;

        View view = LayoutInflater.from(this).inflate(R.layout.dialog_input_history_form, null);

        MaterialButton btnPickImage = view.findViewById(R.id.btn_pick_image);
        TextView tvOriginalUrl = view.findViewById(R.id.tv_original_url);
        TextInputEditText etAlgorithmId = view.findViewById(R.id.et_algorithm_id);
        TextInputEditText etAlgorithmName = view.findViewById(R.id.et_algorithm_name);
        TextInputEditText etAlgorithmParams = view.findViewById(R.id.et_algorithm_params);
        TextInputEditText etProcessingTime = view.findViewById(R.id.et_processing_time);
        Spinner spinnerStatus = view.findViewById(R.id.spinner_status);
        Spinner spinnerSource = view.findViewById(R.id.spinner_source);
        RadioGroup rgFavorite = view.findViewById(R.id.rg_favorite);

        // 保存引用以便上传后更新
        currentOriginalUrlView = tvOriginalUrl;

        ArrayAdapter<String> statusAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, STATUS_LABELS);
        statusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerStatus.setAdapter(statusAdapter);

        ArrayAdapter<String> sourceAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, FORM_SOURCE_LABELS);
        sourceAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerSource.setAdapter(sourceAdapter);

        if (isEdit) {
            if (currentOriginalUrl != null && !currentOriginalUrl.isEmpty()) {
                tvOriginalUrl.setVisibility(View.VISIBLE);
                tvOriginalUrl.setText("已上传: " + currentOriginalUrl);
            }
            if (existing.getAlgorithmId() != null) {
                etAlgorithmId.setText(String.valueOf(existing.getAlgorithmId()));
            }
            etAlgorithmName.setText(safe(existing.getAlgorithmName()));
            etAlgorithmParams.setText(safe(existing.getAlgorithmParams()));
            if (existing.getProcessingTime() != null) {
                etProcessingTime.setText(String.valueOf(existing.getProcessingTime()));
            }
            for (int i = 0; i < STATUS_VALUES.length; i++) {
                if (STATUS_VALUES[i].equals(existing.getStatus())) {
                    spinnerStatus.setSelection(i);
                    break;
                }
            }
            String src = existing.getInputSource();
            int srcIdx = 0;
            for (int i = 0; i < FORM_SOURCE_VALUES.length; i++) {
                if (FORM_SOURCE_VALUES[i].equals(src)) {
                    srcIdx = i;
                    break;
                }
            }
            spinnerSource.setSelection(srcIdx);
            if (existing.getIsFavorite() != null && existing.getIsFavorite() == 1) {
                rgFavorite.check(R.id.rb_favorite_yes);
            } else {
                rgFavorite.check(R.id.rb_favorite_no);
            }
        } else {
            spinnerStatus.setSelection(0);
            spinnerSource.setSelection(0);
            rgFavorite.check(R.id.rb_favorite_no);
        }

        btnPickImage.setOnClickListener(v -> imagePickerLauncher.launch("image/*"));

        new AlertDialog.Builder(this)
                .setTitle(isEdit ? "修改历史记录" : "新增历史记录")
                .setView(view)
                .setPositiveButton("确定", (dialog, which) -> {
                    String algorithmIdStr = getText(etAlgorithmId);
                    String algorithmName = getText(etAlgorithmName);
                    String algorithmParams = getText(etAlgorithmParams);
                    String processingTimeStr = getText(etProcessingTime);
                    Integer status = STATUS_VALUES[spinnerStatus.getSelectedItemPosition()];
                    String source = FORM_SOURCE_VALUES[spinnerSource.getSelectedItemPosition()];
                    Integer isFavorite = rgFavorite.getCheckedRadioButtonId() == R.id.rb_favorite_yes ? 1 : 0;

                    if (isEdit) {
                        InputHistoryUpdateForm form = new InputHistoryUpdateForm();
                        form.setOriginalImageUrl(currentOriginalUrl);
                        if (!TextUtils.isEmpty(algorithmIdStr)) {
                            try {
                                form.setAlgorithmId(Long.parseLong(algorithmIdStr));
                            } catch (NumberFormatException ignored) {
                            }
                        }
                        form.setAlgorithmName(algorithmName);
                        form.setAlgorithmParams(algorithmParams);
                        if (!TextUtils.isEmpty(processingTimeStr)) {
                            try {
                                form.setProcessingTime(Integer.parseInt(processingTimeStr));
                            } catch (NumberFormatException ignored) {
                            }
                        }
                        form.setStatus(status);
                        form.setIsFavorite(isFavorite);
                        viewModel.updateHistory(editingHistoryId, form);
                    } else {
                        if (TextUtils.isEmpty(currentOriginalUrl)) {
                            ToastUtils.showShort(this, "请先选择并上传原始图片");
                            return;
                        }
                        InputHistoryForm form = new InputHistoryForm();
                        form.setOriginalImageUrl(currentOriginalUrl);
                        if (!TextUtils.isEmpty(algorithmIdStr)) {
                            try {
                                form.setAlgorithmId(Long.parseLong(algorithmIdStr));
                            } catch (NumberFormatException ignored) {
                            }
                        }
                        form.setAlgorithmName(algorithmName);
                        form.setAlgorithmParams(algorithmParams);
                        if (!TextUtils.isEmpty(processingTimeStr)) {
                            try {
                                form.setProcessingTime(Integer.parseInt(processingTimeStr));
                            } catch (NumberFormatException ignored) {
                            }
                        }
                        form.setStatus(status);
                        form.setInputSource(source);
                        form.setIsFavorite(isFavorite);
                        viewModel.createHistory(form);
                    }
                    // 重置表单状态
                    editingHistoryId = 0;
                    currentOriginalUrl = null;
                    currentOriginalUrlView = null;
                })
                .setNegativeButton("取消", (dialog, which) -> {
                    editingHistoryId = 0;
                    currentOriginalUrl = null;
                    currentOriginalUrlView = null;
                })
                .setOnDismissListener(dialog -> {
                    editingHistoryId = 0;
                    currentOriginalUrl = null;
                    currentOriginalUrlView = null;
                })
                .show();
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
        tvMessage.setText(safe(result.getMessage()));

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
                .load(url)
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(ivPreview);
        new AlertDialog.Builder(this)
                .setView(view)
                .setPositiveButton("关闭", null)
                .show();
    }

    /**
     * 将 Uri 转换为 File
     */
    private File uriToFile(Uri uri) {
        try {
            InputStream inputStream = getContentResolver().openInputStream(uri);
            if (inputStream == null) return null;
            File tempFile = new File(getCacheDir(), "upload_" + System.currentTimeMillis() + ".jpg");
            try (FileOutputStream outputStream = new FileOutputStream(tempFile)) {
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
            }
            inputStream.close();
            return tempFile;
        } catch (IOException e) {
            return null;
        }
    }

    private String getText(TextInputEditText et) {
        return et.getText() != null ? et.getText().toString().trim() : "";
    }

    private String safe(String s) {
        return s == null ? "" : s;
    }
}
