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
import androidx.appcompat.widget.Toolbar;
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
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.DehazeParams;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.ui.compare.viewmodel.CompareViewModel;
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

    private Toolbar toolbar;
    private ImageView ivSelectedImage;
    private Spinner spinnerAlgorithm;
    private MaterialButton btnAddAlgorithm;
    private TextView tvSelectedAlgorithms;
    private MaterialButton btnCompare;
    private ProgressBar progressBar;
    private ViewPager2 viewPager;
    private TabLayout tabLayout;

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
        setContentView(R.layout.activity_compare);

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        toolbar = findViewById(R.id.toolbar);
        ivSelectedImage = findViewById(R.id.iv_selected_image);
        spinnerAlgorithm = findViewById(R.id.spinner_algorithm);
        btnAddAlgorithm = findViewById(R.id.btn_add_algorithm);
        tvSelectedAlgorithms = findViewById(R.id.tv_selected_algorithms);
        btnCompare = findViewById(R.id.btn_compare);
        progressBar = findViewById(R.id.progress_bar);
        viewPager = findViewById(R.id.view_pager);
        tabLayout = findViewById(R.id.tab_layout);

        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(v -> finish());

        findViewById(R.id.btn_select_image).setOnClickListener(v ->
                pickImageLauncher.launch("image/*"));

        btnAddAlgorithm.setOnClickListener(v -> addCurrentAlgorithm());
        btnCompare.setOnClickListener(v -> onCompareClick());

        viewPager.setAdapter(new ComparePagerAdapter(this));
        new TabLayoutMediator(tabLayout, viewPager,
                (tab, position) -> tab.setText(position == 0 ? "并排对比" : "重叠对比"))
                .attach();
    }

    private void addCurrentAlgorithm() {
        int pos = spinnerAlgorithm.getSelectedItemPosition();
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
        tvSelectedAlgorithms.setText("已选算法：" + TextUtils.join("、", selectedAlgorithmLabels));
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
        compareViewModel.getLoading().observe(this, isLoading ->
                progressBar.setVisibility(isLoading != null && isLoading ? View.VISIBLE : View.GONE));
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
        ivSelectedImage.setVisibility(View.VISIBLE);
        Glide.with(this).load(DehazeSDK.getInstance().resolveUrl(fileInfo.getUrl()))
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(ivSelectedImage);
    }

    private void updateAlgorithmSpinner(List<Option> options) {
        algorithmOptions.clear();
        if (options != null) {
            algorithmOptions.addAll(options);
        }
        ViewUtils.updateAlgorithmSpinner(spinnerAlgorithm, algorithmOptions);
    }

    private void onPredictionResult(PredResult result) {
        if (result == null) return;
        new AlertDialog.Builder(this)
                .setTitle("处理完成")
                .setMessage("耗时：" + (result.getTime() == null ? "-" : result.getTime() + "ms")
                        + (Boolean.TRUE.equals(result.getFromCache()) ? "（命中缓存）" : ""))
                .setPositiveButton("确定", null)
                .show();
    }

    private void onMultiPredictionResults(java.util.Map<Long, PredResult> results) {
        if (results == null) return;
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

    private static class ComparePagerAdapter extends FragmentStateAdapter {

        public ComparePagerAdapter(FragmentActivity fa) {
            super(fa);
        }

        @Override
        public Fragment createFragment(int position) {
            return position == 0 ? new ParallelFragment() : new OverlapFragment();
        }

        @Override
        public int getItemCount() {
            return 2;
        }
    }
}
