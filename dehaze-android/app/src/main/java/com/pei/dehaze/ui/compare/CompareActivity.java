package com.pei.dehaze.ui.compare;

import androidx.appcompat.app.AlertDialog;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import androidx.viewpager2.widget.ViewPager2;

import com.bumptech.glide.Glide;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityCompareBinding;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.DehazeParams;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.ui.compare.viewmodel.CompareViewModel;
import com.pei.dehaze.utils.StatePlaceholder;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;
import com.pei.dehaze.utils.UriUtils;
import com.pei.dehaze.utils.ViewUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class CompareActivity extends AppCompatActivity {

    private CompareViewModel compareViewModel;
    private ActivityCompareBinding binding;
    private StatePlaceholder statePlaceholder;

    private final List<Option> algorithmOptions = new ArrayList<>();
    private final Set<Long> selectedAlgorithmIds = new HashSet<>();
    private final List<String> selectedAlgorithmLabels = new ArrayList<>();

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

        binding.btnAddAlgorithm.setOnClickListener(v -> addCurrentAlgorithm());
        binding.btnCompare.setOnClickListener(v -> onCompareClick());

        binding.viewPager.setAdapter(new ComparePagerAdapter(this));
        new TabLayoutMediator(binding.tabLayout, binding.viewPager,
                (tab, position) -> tab.setText(COMPARE_TABS[position]))
                .attach();

        statePlaceholder = new StatePlaceholder(binding.statePlaceholder.getRoot());
        statePlaceholder.showEmpty("请先上传图片并选择算法", R.drawable.ic_image_compare);
    }

    private void addCurrentAlgorithm() {
        int pos = binding.spinnerAlgorithm.getSelectedItemPosition();
        if (pos < 0 || pos >= algorithmOptions.size()) {
            ToastUtils.showShort(this, "请先选择算法");
            return;
        }
        Option option = algorithmOptions.get(pos);
        Long id = option.getValue() == null ? null : StringUtils.safeParseLong(option.getValue(), 0L);
        if (id == null) {
            ToastUtils.showShort(this, "算法ID无效");
            return;
        }
        if (selectedAlgorithmIds.contains(id)) {
            ToastUtils.showShort(this, "该算法已添加");
            return;
        }
        selectedAlgorithmIds.add(id);
        selectedAlgorithmLabels.add(option.getLabel());
        binding.tvSelectedAlgorithms.setText("已选算法：" + TextUtils.join("、", selectedAlgorithmLabels));
    }

    private void onCompareClick() {
        if (compareViewModel.getOriginalImageUrl() == null) {
            ToastUtils.showShort(this, "请先上传图片");
            return;
        }
        if (selectedAlgorithmIds.isEmpty()) {
            ToastUtils.showShort(this, "请至少添加一个算法");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("确认对比")
                .setMessage("确认开始对比处理？将使用 " + selectedAlgorithmIds.size() + " 个算法处理图片。")
                .setPositiveButton("确定", (d, w) -> {
                    DehazeParams params = new DehazeParams();
                    if (selectedAlgorithmIds.size() == 1) {
                        Long id = selectedAlgorithmIds.iterator().next();
                        compareViewModel.predict(id, params);
                    } else {
                        compareViewModel.predictMultiple(new ArrayList<>(selectedAlgorithmIds), params);
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
        compareViewModel.getAlgorithmOptions().observe(this, this::updateAlgorithmSpinner);
        compareViewModel.getPredictionResult().observe(this, this::onPredictionResult);
        compareViewModel.getMultiPredictionResults().observe(this, this::onMultiPredictionResults);
        compareViewModel.getLoading().observe(this, isLoading -> {
            boolean loading = isLoading != null && isLoading;
            binding.progressBar.setVisibility(loading ? View.VISIBLE : View.GONE);
            if (loading) {
                statePlaceholder.showLoading("正在处理中…");
            }
        });
        compareViewModel.getError().observe(this, errorMessage -> {
            if (!TextUtils.isEmpty(errorMessage)) {
                ToastUtils.showShort(this, errorMessage);
                compareViewModel.clearError();
            }
        });
        compareViewModel.getOperationResult().observe(this, result -> {
            if (!TextUtils.isEmpty(result)) {
                ToastUtils.showShort(this, result);
                compareViewModel.clearOperationResult();
            }
        });
    }

    private void showUploadedImage(FileInfo fileInfo) {
        if (fileInfo == null || fileInfo.getUrl() == null) return;
        binding.ivSelectedImage.setVisibility(View.VISIBLE);
        Glide.with(this).load(fileInfo.getUrl())
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(binding.ivSelectedImage);
    }

    private void updateAlgorithmSpinner(List<Option> options) {
        algorithmOptions.clear();
        if (options != null) {
            algorithmOptions.addAll(options);
        }
        ViewUtils.updateAlgorithmSpinner(binding.spinnerAlgorithm, algorithmOptions);
    }

    private void onPredictionResult(PredResult result) {
        if (result == null) return;
        statePlaceholder.hide();
        new AlertDialog.Builder(this)
                .setTitle("处理完成")
                .setMessage("耗时：" + (result.getTime() == null ? "-" : result.getTime() + "ms"))
                .setPositiveButton("确定", null)
                .show();
    }

    private void onMultiPredictionResults(java.util.Map<Long, PredResult> results) {
        if (results == null || results.isEmpty()) return;
        statePlaceholder.hide();
        int success = results.size();
        int total = selectedAlgorithmIds.size();
        ToastUtils.showShort(this, "完成 " + success + "/" + total + " 个算法处理");
    }

    private void loadData() {
        compareViewModel.loadAlgorithmOptions();
    }

    private void uploadImage(Uri uri) {
        File tempFile = UriUtils.copyToCache(this, uri);
        if (tempFile == null) {
            ToastUtils.showShort(this, "无法读取所选图片");
            return;
        }
        compareViewModel.uploadImage(tempFile);
    }

    /** 6 种对比模式（设计稿要求） */
    private static final String[] COMPARE_TABS = {
            "并排对比", "重叠对比", "放大镜", "滤镜调节", "指标评估", "算法信息"
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
                case 5: return new AlgorithmInfoFragment();
                default: return new ParallelFragment();
            }
        }

        @Override
        public int getItemCount() {
            return COMPARE_TABS.length;
        }
    }
}
