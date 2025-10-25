package com.pei.dehaze.ui.dataset;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.RecyclerView;
import androidx.recyclerview.widget.StaggeredGridLayoutManager;

import com.google.android.material.button.MaterialButton;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentDatasetDetailBinding;
import com.pei.dehaze.ui.dataset.model.ImageType;
import com.pei.dehaze.ui.dataset.model.ViewCard;

import java.util.List;

public class DatasetDetailFragment extends Fragment {

    private static final String ARG_DATASET_ID = "dataset_id";

    private FragmentDatasetDetailBinding binding;
    private DatasetViewModel viewModel;
    private DatasetImageAdapter imageAdapter;
    private int datasetId;

    public static DatasetDetailFragment newInstance(int datasetId) {
        DatasetDetailFragment fragment = new DatasetDetailFragment();
        Bundle args = new Bundle();
        args.putInt(ARG_DATASET_ID, datasetId);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            datasetId = getArguments().getInt(ARG_DATASET_ID);
        }
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentDatasetDetailBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // 初始化 ViewModel
        viewModel = new ViewModelProvider(this).get(DatasetViewModel.class);
        viewModel.setDatasetId(datasetId);

        // 设置数据绑定
        binding.setViewModel(viewModel);
        binding.setLifecycleOwner(this);

        // 初始化界面
        setupUI();

        // 加载数据
        viewModel.loadDatasetInfo();
        viewModel.loadImages();
    }

    private void setupUI() {
        // 设置图片瀑布流
        StaggeredGridLayoutManager layoutManager = new StaggeredGridLayoutManager(
                2, StaggeredGridLayoutManager.VERTICAL);
        binding.recyclerView.setLayoutManager(layoutManager);
        
        imageAdapter = new DatasetImageAdapter(new DatasetImageAdapter.OnItemClickListener() {
            @Override
            public void onItemClick(ViewCard image, int position) {
                // TODO: 实现大图查看功能
                Toast.makeText(getContext(), "点击图片: " + image.getId(), Toast.LENGTH_SHORT).show();
            }
        });
        binding.recyclerView.setAdapter(imageAdapter);

        // 观察图片数据变化
        viewModel.getImages().observe(getViewLifecycleOwner(), images -> {
            imageAdapter.submitList(images);
        });

        // 观察图片类型变化
        viewModel.getImageTypes().observe(getViewLifecycleOwner(), this::setupImageTypeButtons);

        // 观察错误信息
        viewModel.getError().observe(getViewLifecycleOwner(), error -> {
            if (!error.isEmpty()) {
                Toast.makeText(getContext(), error, Toast.LENGTH_LONG).show();
            }
        });

        // 设置刷新监听
        binding.swipeRefreshLayout.setOnRefreshListener(() -> {
            viewModel.resetQuery();
            binding.swipeRefreshLayout.setRefreshing(false);
        });

        // 设置搜索按钮点击事件
        binding.searchButton.setOnClickListener(v -> {
            String keywords = binding.searchEditText.getText().toString().trim();
            viewModel.searchImages(keywords);
        });

        // 设置重置按钮点击事件
        binding.resetButton.setOnClickListener(v -> {
            binding.searchEditText.setText("");
            viewModel.resetQuery();
        });
    }

    private void setupImageTypeButtons(List<ImageType> imageTypes) {
        // 清除现有的按钮
        binding.buttonContainer.removeAllViews();

        // 为每个图片类型创建按钮
        for (ImageType type : imageTypes) {
            MaterialButton button = new MaterialButton(getContext(), null, 
                    com.google.android.material.R.attr.materialButtonStyle);
            button.setText(type.getType());
            button.setTag(type.getId());
            
            // 设置按钮样式
            if (type.isEnabled()) {
                button.setBackgroundColor(getResources().getColor(
                        com.google.android.material.R.color.design_default_color_primary, 
                        getContext().getTheme()));
            } else {
                // 使用透明背景
                button.setBackgroundColor(
                        getResources().getColor(android.R.color.transparent, getContext().getTheme()));
            }
            
            button.setOnClickListener(v -> {
                int typeId = (int) v.getTag();
                viewModel.switchImageType(typeId);
            });
            
            // 添加按钮到容器
            binding.buttonContainer.addView(button);
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}