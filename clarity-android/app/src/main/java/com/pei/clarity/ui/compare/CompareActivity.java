package com.pei.clarity.ui.compare;

import androidx.appcompat.app.AlertDialog;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import com.pei.clarity.ui.common.BaseActivity;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import androidx.viewpager2.widget.ViewPager2;

import com.bumptech.glide.Glide;
import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;
import com.pei.clarity.R;
import com.pei.clarity.databinding.ActivityCompareBinding;
import com.pei.clarity.sdk.model.ai_model.AiModelVO;
import com.pei.clarity.sdk.model.file.FileInfo;
import com.pei.clarity.sdk.model.image_task.DehazeParams;
import com.pei.clarity.sdk.model.image_task.ImageTaskResponse;
import com.pei.clarity.ui.compare.viewmodel.CompareViewModel;
import com.pei.clarity.utils.StatePlaceholder;
import com.pei.clarity.utils.ToastUtils;
import com.pei.clarity.utils.UriUtils;
import com.pei.clarity.utils.ViewUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class CompareActivity extends BaseActivity {

    private CompareViewModel compareViewModel;
    private ActivityCompareBinding binding;
    private StatePlaceholder statePlaceholder;

    private final List<AiModelVO> modelOptions = new ArrayList<>();
    private final Set<String> selectedModelIds = new HashSet<>();
    private final List<String> selectedModelLabels = new ArrayList<>();

    private final ActivityResultLauncher<String> pickImageLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    uploadImage(uri);
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityCompareBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        setSupportActionBar(binding.toolbar);
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        binding.btnSelectImage.setOnClickListener(v ->
                pickImageLauncher.launch("image/*"));

        binding.btnAddModel.setOnClickListener(v -> addCurrentModel());
        binding.btnCompare.setOnClickListener(v -> onCompareClick());

        binding.viewPager.setAdapter(new ComparePagerAdapter(this));
        new TabLayoutMediator(binding.tabLayout, binding.viewPager,
                (tab, position) -> tab.setText(COMPARE_TABS[position]))
                .attach();

        statePlaceholder = new StatePlaceholder(binding.statePlaceholder.getRoot());
        statePlaceholder.showEmpty("请先上传图片并选择模型", R.drawable.ic_image_compare);
    }

    private void addCurrentModel() {
        int pos = binding.spinnerModel.getSelectedItemPosition();
        if (pos < 0 || pos >= modelOptions.size()) {
            ToastUtils.showShort(this, "请先选择模型");
            return;
        }
        AiModelVO model = modelOptions.get(pos);
        String modelId = model.getModelId();
        if (modelId == null) {
            ToastUtils.showShort(this, "模型标识无效");
            return;
        }
        if (selectedModelIds.contains(modelId)) {
            ToastUtils.showShort(this, "该模型已添加");
            return;
        }
        selectedModelIds.add(modelId);
        selectedModelLabels.add(model.getDisplayName() != null ? model.getDisplayName() : modelId);
        binding.tvSelectedModels.setText("已选模型：" + TextUtils.join("、", selectedModelLabels));
    }

    private void onCompareClick() {
        if (compareViewModel.getOriginalImageUrl() == null) {
            ToastUtils.showShort(this, "请先上传图片");
            return;
        }
        if (selectedModelIds.isEmpty()) {
            ToastUtils.showShort(this, "请至少添加一个模型");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("确认对比")
                .setMessage("确认开始对比处理？将使用 " + selectedModelIds.size() + " 个模型处理图片。")
                .setPositiveButton("确定", (d, w) -> {
                    DehazeParams params = new DehazeParams();
                    if (selectedModelIds.size() == 1) {
                        compareViewModel.predict(selectedModelIds.iterator().next(), params);
                    } else {
                        compareViewModel.predictMultiple(new ArrayList<>(selectedModelIds), params);
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void initViewModel() {
        compareViewModel = new ViewModelProvider(this).get(CompareViewModel.class);
    }

    private void setupObservers() {
        compareViewModel.getUploadedFile().observe(this, this::showUploadedImage);
        compareViewModel.getModels().observe(this, this::updateModelSpinner);
        compareViewModel.getPredictionResult().observe(this, this::onPredictionResult);
        compareViewModel.getMultiPredictionResults().observe(this, this::onMultiPredictionResults);
        compareViewModel.getLoading().observe(this, isLoading -> {
            boolean loading = isLoading != null && isLoading;
            binding.progressBar.setVisibility(loading ? View.VISIBLE : View.GONE);
            if (loading) {
                statePlaceholder.showLoading("正在处理中…");
            }
        });
        observeError(compareViewModel);
        observeOperationResult(compareViewModel, null);
    }

    private void showUploadedImage(FileInfo fileInfo) {
        if (fileInfo == null || fileInfo.getUrl() == null) return;
        binding.ivSelectedImage.setVisibility(View.VISIBLE);
        Glide.with(this).load(fileInfo.getUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(binding.ivSelectedImage);
    }

    private void updateModelSpinner(List<AiModelVO> models) {
        modelOptions.clear();
        if (models != null) {
            modelOptions.addAll(models);
        }
        ViewUtils.updateModelSpinner(binding.spinnerModel, modelOptions);
    }

    private void onPredictionResult(ImageTaskResponse result) {
        if (result == null) return;
        statePlaceholder.hide();
        new AlertDialog.Builder(this)
                .setTitle("处理完成")
                .setMessage("耗时：" + (result.getTime() == null ? "-" : result.getTime() + "s"))
                .setPositiveButton("确定", null)
                .show();
    }

    private void onMultiPredictionResults(java.util.Map<String, ImageTaskResponse> results) {
        if (results == null || results.isEmpty()) return;
        statePlaceholder.hide();
        int success = results.size();
        int total = selectedModelIds.size();
        ToastUtils.showShort(this, "完成 " + success + "/" + total + " 个模型处理");
    }

    private void loadData() {
        compareViewModel.loadModels();
    }

    private void uploadImage(Uri uri) {
        File tempFile = UriUtils.copyToCache(this, uri);
        if (tempFile == null) {
            ToastUtils.showShort(this, "无法读取所选图片");
            return;
        }
        compareViewModel.uploadImage(tempFile);
    }

    /** 5 种对比模式 */
    private static final String[] COMPARE_TABS = {
            "并排对比", "重叠对比", "放大镜", "滤镜调节", "指标评估"
    };

    private static class ComparePagerAdapter extends FragmentStateAdapter {

        public ComparePagerAdapter(FragmentActivity fa) {
            super(fa);
        }

        @Override
        public Fragment createFragment(int position) {
            switch (position) {
                case 0: return new ParallelFragment();
                case 1: return new OverlapFragment();
                case 2: return new MagnifierFragment();
                case 3: return new FilterFragment();
                case 4: return new MetricsFragment();
                default: return new ParallelFragment();
            }
        }

        @Override
        public int getItemCount() {
            return COMPARE_TABS.length;
        }
    }
}
