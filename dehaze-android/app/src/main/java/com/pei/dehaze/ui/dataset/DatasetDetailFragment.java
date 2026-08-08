package com.pei.dehaze.ui.dataset;

import android.graphics.Color;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.GridLayoutManager;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentDatasetDetailBinding;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.DatasetStatistics;
import com.pei.dehaze.sdk.model.dataset.ImageItem;
import com.pei.dehaze.sdk.model.dataset.ImageType;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;

/**
 * 数据集详情浏览版（仅查看图片网格 + 分页，无增删改操作）
 */
public class DatasetDetailFragment extends Fragment {

    private static final String ARG_DATASET_ID = "dataset_id";

    private DatasetDetailViewModel viewModel;
    private DatasetImageAdapter imageAdapter;
    private FragmentDatasetDetailBinding binding;
    private long datasetId;

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

        initViews();
        setupObservers();
        viewModel.refresh();
    }

    private void initViews() {
        binding.toolbar.setNavigationOnClickListener(v ->
                requireActivity().getOnBackPressedDispatcher().onBackPressed());

        imageAdapter = new DatasetImageAdapter();
        imageAdapter.setShowActions(false); // 浏览版无编辑/删除/上传操作
        binding.recyclerView.setLayoutManager(new GridLayoutManager(getContext(), 2));
        binding.recyclerView.setAdapter(imageAdapter);

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
                // 图片点击预览（由 item 自带处理）
            }

            @Override
            public void onEdit(ImageItem item) { }

            @Override
            public void onDelete(ImageItem item) { }

            @Override
            public void onUploadFile(ImageItem item) { }

            @Override
            public void onDeleteFile(ImageItem item, com.pei.dehaze.sdk.model.dataset.ImageUrl url) { }
        });

        binding.btnPrev.setOnClickListener(v -> viewModel.prevPage());
        binding.btnNext.setOnClickListener(v -> viewModel.nextPage());
    }

    private void setupObservers() {
        viewModel.getDatasetInfo().observe(getViewLifecycleOwner(), this::bindDatasetInfo);

        viewModel.getItems().observe(getViewLifecycleOwner(), items -> {
            imageAdapter.submitList(items);
            updatePageInfo();
            binding.tvEmpty.setVisibility(items == null || items.isEmpty() ? View.VISIBLE : View.GONE);
        });

        viewModel.getTotal().observe(getViewLifecycleOwner(), total -> updatePageInfo());

        viewModel.getError().observe(getViewLifecycleOwner(), errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(getContext(), errorMessage);
                viewModel.clearError();
            }
        });
    }

    private void bindDatasetInfo(Dataset dataset) {
        if (dataset == null) return;
        binding.tvDatasetName.setText(StringUtils.safe(dataset.getName()));

        Integer status = dataset.getStatus();
        if (status != null && status == 1) {
            binding.tvDatasetStatus.setText("公开");
            binding.tvDatasetStatus.setTextColor(Color.parseColor("#4CAF50"));
        } else {
            binding.tvDatasetStatus.setText("私有");
            binding.tvDatasetStatus.setTextColor(Color.parseColor("#9E9E9E"));
        }

        binding.tvDatasetType.setText("类型: " + StringUtils.safe(dataset.getType()));
        binding.tvDatasetDescription.setText(StringUtils.safe(dataset.getDescription()));

        DatasetStatistics stats = dataset.getStatistics();
        if (stats != null) {
            binding.tvStatItems.setText("数据项: " + (stats.getItemCount() != null ? stats.getItemCount() : 0));
            binding.tvStatFiles.setText("文件: " + (stats.getFileCount() != null ? stats.getFileCount() : 0));
        }
    }

    private void updatePageInfo() {
        long totalVal = viewModel.getTotal().getValue() != null ? viewModel.getTotal().getValue() : 0L;
        int pageNum = viewModel.getPageNum();
        int pageSize = viewModel.getPageSize();
        int totalPages = Math.max(1, (int) Math.ceil(totalVal * 1.0 / pageSize));
        binding.tvPageInfo.setText("第 " + pageNum + " 页 / 共 " + totalPages + " 页 (共 " + totalVal + " 条)");
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
