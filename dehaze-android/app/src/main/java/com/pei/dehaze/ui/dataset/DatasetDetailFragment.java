package com.pei.dehaze.ui.dataset;

import androidx.appcompat.app.AlertDialog;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.RadioGroup;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.bumptech.glide.Glide;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.button.MaterialButtonToggleGroup;
import com.google.android.material.textfield.TextInputEditText;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentDatasetDetailBinding;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.UriUtils;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetStatistics;
import com.pei.dehaze.sdk.model.dataset.ImageItem;
import com.pei.dehaze.sdk.model.dataset.ImageType;
import com.pei.dehaze.sdk.model.dataset.ImageUrl;
import com.pei.dehaze.utils.ToastUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * 数据集详情 Fragment（数据集元数据 + 数据项分页 + 数据项 CRUD + 图片文件管理 + 图片类型切换）
 */
public class DatasetDetailFragment extends Fragment {

    private static final String ARG_DATASET_ID = "dataset_id";

    private DatasetDetailViewModel viewModel;
    private DatasetImageAdapter imageAdapter;
    private FragmentDatasetDetailBinding binding;

    private long datasetId;

    /** 待上传图片的目标数据项ID */
    private long pendingUploadItemId = 0;
    /** 已选择的待上传文件 Uri */
    private Uri pendingUploadUri;

    private ActivityResultLauncher<String> imagePickerLauncher;

    public static DatasetDetailFragment newInstance(long datasetId) {
        DatasetDetailFragment fragment = new DatasetDetailFragment();
        Bundle args = new Bundle();
        args.putLong(ARG_DATASET_ID, datasetId);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            datasetId = getArguments().getLong(ARG_DATASET_ID, 0);
        }

        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri == null) {
                        ToastUtils.showShort(getContext(), "未选择图片");
                        return;
                    }
                    pendingUploadUri = uri;
                    showUploadDialog(pendingUploadItemId);
                });
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentDatasetDetailBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        viewModel = new ViewModelProvider(this).get(DatasetDetailViewModel.class);
        viewModel.setDatasetId(datasetId);

        bindViews();
        setupListeners();
        setupObservers();

        viewModel.refresh();
    }

    private void bindViews() {
        binding.toolbar.setNavigationOnClickListener(v -> requireActivity().getOnBackPressedDispatcher().onBackPressed());

        imageAdapter = new DatasetImageAdapter();
        binding.recyclerView.setLayoutManager(new GridLayoutManager(getContext(), 2));
        binding.recyclerView.setAdapter(imageAdapter);

        // 默认选中 hazy
        binding.toggleImageType.check(R.id.btn_type_hazy);
    }

    private void setupListeners() {
        binding.swipeRefresh.setOnRefreshListener(() -> viewModel.refresh());

        binding.btnSearch.setOnClickListener(v -> {
            String kw = binding.etKeywords.getText().toString().trim();
            viewModel.setQueryParams(kw, null, null);
            viewModel.loadItems();
        });

        binding.btnReset.setOnClickListener(v -> {
            binding.etKeywords.setText("");
            viewModel.resetQuery();
            viewModel.loadItems();
        });

        binding.btnAddItem.setOnClickListener(v -> showItemFormDialog(null));

        binding.btnBatchDelete.setOnClickListener(v -> {
            if (!imageAdapter.isSelectionMode()) {
                imageAdapter.setSelectionMode(true);
                updateSelectionUI(true);
                ToastUtils.showShort(getContext(), "请勾选要删除的数据项");
            } else {
                confirmBatchDeleteItems();
            }
        });

        binding.btnCancelSelect.setOnClickListener(v -> {
            imageAdapter.clearSelection();
            imageAdapter.setSelectionMode(false);
            updateSelectionUI(false);
        });

        binding.btnSelectAll.setOnClickListener(v -> imageAdapter.selectAll());

        binding.btnPrev.setOnClickListener(v -> viewModel.prevPage());
        binding.btnNext.setOnClickListener(v -> viewModel.nextPage());

        binding.toggleImageType.addOnButtonCheckedListener((group, checkedId, isChecked) -> {
            if (!isChecked) return;
            ImageType type;
            if (checkedId == R.id.btn_type_clear) {
                type = ImageType.CLEAR;
            } else if (checkedId == R.id.btn_type_hazy) {
                type = ImageType.HAZY;
            } else if (checkedId == R.id.btn_type_trans) {
                type = ImageType.TRANS;
            } else {
                return;
            }
            viewModel.setCurrentImageType(type);
            imageAdapter.setCurrentImageType(type);
        });

        imageAdapter.setActionListener(new DatasetImageAdapter.OnItemActionListener() {
            @Override
            public void onItemClick(ImageItem item, String imageUrl) {
                showImagePreviewDialog(imageUrl);
            }

            @Override
            public void onEdit(ImageItem item) {
                showItemFormDialog(item);
            }

            @Override
            public void onDelete(ImageItem item) {
                confirmDeleteItem(item);
            }

            @Override
            public void onUploadFile(ImageItem item) {
                if (item.getId() == null) {
                    ToastUtils.showShort(getContext(), "数据项ID无效");
                    return;
                }
                pendingUploadItemId = item.getId();
                imagePickerLauncher.launch("image/*");
            }

            @Override
            public void onDeleteFile(ImageItem item, ImageUrl url) {
                confirmDeleteFile(url);
            }
        });

        imageAdapter.setSelectionListener(selectedIds ->
                binding.btnBatchDelete.setText(selectedIds.isEmpty() ? "批量删除" : "删除选中(" + selectedIds.size() + ")"));
    }

    private void setupObservers() {
        viewModel.getDatasetInfo().observe(getViewLifecycleOwner(), this::bindDatasetInfo);

        viewModel.getItems().observe(getViewLifecycleOwner(), items -> {
            imageAdapter.submitList(items);
            updatePageInfo();
            binding.tvEmpty.setVisibility(items == null || items.isEmpty() ? View.VISIBLE : View.GONE);
        });

        viewModel.getTotal().observe(getViewLifecycleOwner(), total -> updatePageInfo());

        viewModel.getLoading().observe(getViewLifecycleOwner(), isLoading ->
                binding.swipeRefresh.setRefreshing(isLoading != null && isLoading));

        viewModel.getError().observe(getViewLifecycleOwner(), errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(getContext(), errorMessage);
                viewModel.clearError();
            }
        });

        viewModel.getOperationResult().observe(getViewLifecycleOwner(), result -> {
            if (result != null && !result.isEmpty()) {
                ToastUtils.showShort(getContext(), result);
                viewModel.clearOperationResult();
                if (result.startsWith("删除") && imageAdapter.isSelectionMode()) {
                    imageAdapter.clearSelection();
                    imageAdapter.setSelectionMode(false);
                    updateSelectionUI(false);
                }
            }
        });
    }

    private void bindDatasetInfo(Dataset dataset) {
        if (dataset == null) return;
        binding.tvDatasetName.setText(StringUtils.safe(dataset.getName()));
        Integer status = dataset.getStatus();
        if (status != null && status == 1) {
            binding.tvDatasetStatus.setText("启用");
            binding.tvDatasetStatus.setTextColor(Color.parseColor("#4CAF50"));
        } else {
            binding.tvDatasetStatus.setText("禁用");
            binding.tvDatasetStatus.setTextColor(Color.parseColor("#9E9E9E"));
        }
        binding.tvDatasetType.setText("类型: " + StringUtils.safe(dataset.getType()));
        binding.tvDatasetPath.setText("路径: " + StringUtils.safe(dataset.getPath()));
        binding.tvDatasetDescription.setText(StringUtils.safe(dataset.getDescription()));

        DatasetStatistics stats = dataset.getStatistics();
        if (stats != null) {
            binding.tvStatItems.setText("数据项: " + (stats.getItemCount() != null ? stats.getItemCount() : 0));
            binding.tvStatFiles.setText("文件: " + (stats.getFileCount() != null ? stats.getFileCount() : 0));
            binding.tvStatSize.setText("大小: " + (stats.getTotalSize() != null ? stats.getTotalSize() : 0));
            binding.tvStatClear.setText("已标注: " + (stats.getAnnotatedCount() != null ? stats.getAnnotatedCount() : 0));
            binding.tvStatHazy.setText("未标注: " + (stats.getUnannotatedCount() != null ? stats.getUnannotatedCount() : 0));
            binding.tvStatDistribution.setText(formatDistribution(stats));
        } else {
            binding.tvStatItems.setText("数据项: 0");
            binding.tvStatFiles.setText("文件: 0");
            binding.tvStatSize.setText("大小: 0");
            binding.tvStatClear.setText("清晰: 0");
            binding.tvStatHazy.setText("雾化: 0");
            binding.tvStatDistribution.setText("");
        }
    }

    private String formatDistribution(DatasetStatistics stats) {
        StringBuilder sb = new StringBuilder();
        if (stats.getSceneDistribution() != null && !stats.getSceneDistribution().isEmpty()) {
            sb.append("场景: ");
            for (java.util.Map.Entry<String, Long> entry : stats.getSceneDistribution().entrySet()) {
                sb.append(entry.getKey()).append("=").append(entry.getValue()).append(" ");
            }
            sb.append("\n");
        }
        if (stats.getHazeDistribution() != null && !stats.getHazeDistribution().isEmpty()) {
            sb.append("雾霾程度: ");
            for (java.util.Map.Entry<String, Long> entry : stats.getHazeDistribution().entrySet()) {
                sb.append(entry.getKey()).append("=").append(entry.getValue()).append(" ");
            }
        }
        return sb.toString().trim();
    }

    private void updateSelectionUI(boolean selectionMode) {
        binding.btnCancelSelect.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        binding.btnSelectAll.setVisibility(selectionMode ? View.VISIBLE : View.GONE);
        binding.btnAddItem.setVisibility(selectionMode ? View.GONE : View.VISIBLE);
        binding.btnBatchDelete.setText(selectionMode ? "删除选中" : "批量删除");
    }

    private void updatePageInfo() {
        if (imageAdapter != null && imageAdapter.isSelectionMode()) {
            int count = imageAdapter.getSelectedIds().size();
            binding.tvPageInfo.setText("已选中 " + count + " 项");
            return;
        }
        long totalVal = viewModel.getTotal().getValue() != null ? viewModel.getTotal().getValue() : 0L;
        int pageNum = viewModel.getPageNum();
        int pageSize = viewModel.getPageSize();
        int totalPages = Math.max(1, (int) Math.ceil(totalVal * 1.0 / pageSize));
        binding.tvPageInfo.setText("第 " + pageNum + " 页 / 共 " + totalPages + " 页 (共 " + totalVal + " 条)");
    }

    private void showItemFormDialog(ImageItem existing) {
        boolean isEdit = existing != null && existing.getId() != null;
        View formView = LayoutInflater.from(requireContext()).inflate(R.layout.dialog_dataset_item_form, null);
        TextInputEditText etName = formView.findViewById(R.id.et_name);
        if (isEdit) {
            etName.setText(StringUtils.safe(existing.getName()));
        }

        new AlertDialog.Builder(requireContext())
                .setTitle(isEdit ? "修改数据项" : "新增数据项")
                .setView(formView)
                .setPositiveButton("确定", (dialog, which) -> {
                    String name = etName.getText() != null ? etName.getText().toString().trim() : "";
                    if (TextUtils.isEmpty(name)) {
                        ToastUtils.showShort(getContext(), "数据项名称不能为空");
                        return;
                    }
                    if (isEdit) {
                        viewModel.updateItem(existing.getId(), name);
                    } else {
                        viewModel.createItem(name);
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void confirmDeleteItem(ImageItem item) {
        new AlertDialog.Builder(requireContext())
                .setTitle("删除确认")
                .setMessage("确认删除数据项「" + StringUtils.safe(item.getName()) + "」吗？此操作将同时删除关联的图片文件，且不可恢复！")
                .setPositiveButton("确定", (dialog, which) -> {
                    if (item.getId() != null) {
                        viewModel.deleteItem(item.getId());
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void confirmBatchDeleteItems() {
        Set<Long> selectedIds = imageAdapter.getSelectedIds();
        if (selectedIds.isEmpty()) {
            ToastUtils.showShort(getContext(), "请先选择要删除的数据项");
            return;
        }
        new AlertDialog.Builder(requireContext())
                .setTitle("批量删除确认")
                .setMessage("确认删除选中的 " + selectedIds.size() + " 个数据项吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) ->
                        viewModel.batchDeleteItems(new ArrayList<>(selectedIds)))
                .setNegativeButton("取消", null)
                .show();
    }

    private void confirmDeleteFile(ImageUrl url) {
        if (url.getId() == null) {
            ToastUtils.showShort(getContext(), "图片ID无效");
            return;
        }
        new AlertDialog.Builder(requireContext())
                .setTitle("删除图片确认")
                .setMessage("确认删除该图片吗？删除后不可恢复。")
                .setPositiveButton("确定", (dialog, which) ->
                        viewModel.deleteItemFile(url.getId()))
                .setNegativeButton("取消", null)
                .show();
    }

    /**
     * 显示图片上传对话框（用户选择类型 + 描述，并触发上传）
     */
    private void showUploadDialog(long datasetItemId) {
        if (pendingUploadUri == null) {
            ToastUtils.showShort(getContext(), "未选择图片");
            return;
        }
        View formView = LayoutInflater.from(requireContext()).inflate(R.layout.dialog_item_file_upload, null);
        RadioGroup rgType = formView.findViewById(R.id.rg_type);
        TextInputEditText etDescription = formView.findViewById(R.id.et_description);
        TextView tvFileName = formView.findViewById(R.id.tv_file_name);
        MaterialButton btnPickFile = formView.findViewById(R.id.btn_pick_file);

        // 当前选中的类型默认
        rgType.check(R.id.rb_type_hazy);
        tvFileName.setVisibility(View.VISIBLE);
        tvFileName.setText("已选择: " + pendingUploadUri.getLastPathSegment());

        btnPickFile.setOnClickListener(v -> imagePickerLauncher.launch("image/*"));

        new AlertDialog.Builder(requireContext())
                .setTitle("上传图片")
                .setView(formView)
                .setPositiveButton("上传", (dialog, which) -> {
                    ImageType type;
                    int checkedId = rgType.getCheckedRadioButtonId();
                    if (checkedId == R.id.rb_type_clear) {
                        type = ImageType.CLEAR;
                    } else if (checkedId == R.id.rb_type_trans) {
                        type = ImageType.TRANS;
                    } else {
                        type = ImageType.HAZY;
                    }
                    String description = etDescription.getText() != null
                            ? etDescription.getText().toString().trim() : "";

                    File file = UriUtils.copyToCache(requireContext(), pendingUploadUri);
                    if (file == null) {
                        ToastUtils.showShort(getContext(), "无法读取选择的图片文件");
                        return;
                    }
                    viewModel.uploadItemFile(datasetItemId, type, file, description);
                    pendingUploadUri = null;
                })
                .setNegativeButton("取消", (dialog, which) -> pendingUploadUri = null)
                .show();
    }

    /**
     * 图片预览对话框
     */
    private void showImagePreviewDialog(String imageUrl) {
        View view = LayoutInflater.from(requireContext()).inflate(R.layout.dialog_image_preview, null);
        ImageView ivPreview = view.findViewById(R.id.iv_preview);
        Glide.with(requireContext())
                .load(imageUrl)
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(ivPreview);
        new AlertDialog.Builder(requireContext())
                .setView(view)
                .setPositiveButton("关闭", null)
                .show();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }

}

