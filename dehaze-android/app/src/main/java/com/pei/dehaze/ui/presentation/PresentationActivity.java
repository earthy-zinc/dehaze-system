package com.pei.dehaze.ui.presentation;

import androidx.appcompat.app.AlertDialog;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.SeekBar;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.viewpager2.adapter.FragmentStateAdapter;

import com.bumptech.glide.Glide;
import com.google.android.material.tabs.TabLayoutMediator;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityPresentationBinding;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.DehazeParams;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.ui.common.adapter.PredictionLogAdapter;
import com.pei.dehaze.ui.presentation.viewmodel.PresentationViewModel;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;
import com.pei.dehaze.utils.UriUtils;
import com.pei.dehaze.utils.ViewUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;

public class PresentationActivity extends AppCompatActivity {

    private PresentationViewModel presentationViewModel;
    private ActivityPresentationBinding binding;

    private final List<Option> algorithmOptions = new ArrayList<>();
    private long presetAlgorithmId = 0L;
    private PredictionLogAdapter historyAdapter;
    private ResultPagerAdapter pagerAdapter;

    private final ActivityResultLauncher<String> pickImageLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) uploadImage(uri);
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityPresentationBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        presetAlgorithmId = getIntent().getLongExtra("algorithm_id", 0L);
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

        binding.seekBarStrength.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                binding.tvStrengthValue.setText("强度：" + progress);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        binding.btnPredict.setOnClickListener(v -> onPredictClick());

        pagerAdapter = new ResultPagerAdapter(this);
        binding.viewPager.setAdapter(pagerAdapter);
        new TabLayoutMediator(binding.tabLayout, binding.viewPager,
                (tab, position) -> tab.setText(position == 0 ? "原图" : "去雾结果"))
                .attach();

        historyAdapter = new PredictionLogAdapter();
        binding.rvHistory.setLayoutManager(new LinearLayoutManager(this));
        binding.rvHistory.setAdapter(historyAdapter);
    }

    private void onPredictClick() {
        Long algorithmId = getCurrentAlgorithmId();
        if (algorithmId == null) {
            ToastUtils.showShort(this, "请先选择算法");
            return;
        }
        int strength = binding.seekBarStrength.getProgress();
        DehazeParams params = new DehazeParams(strength, 100, 100, 30);
        new AlertDialog.Builder(this)
                .setTitle("确认处理")
                .setMessage("确认开始去雾处理？")
                .setPositiveButton("确定", (d, w) ->
                        presentationViewModel.predict(algorithmId, params))
                .setNegativeButton("取消", null)
                .show();
    }

    private Long getCurrentAlgorithmId() {
        int pos = binding.spinnerAlgorithm.getSelectedItemPosition();
        if (pos < 0 || pos >= algorithmOptions.size()) return null;
        Option option = algorithmOptions.get(pos);
        return option.getValue() == null ? null : StringUtils.safeParseLong(option.getValue(), 0L);
    }

    private void initViewModel() {
        presentationViewModel = new ViewModelProvider(this).get(PresentationViewModel.class);
    }

    private void setupObservers() {
        presentationViewModel.getUploadedFile().observe(this, this::showOriginalImage);
        presentationViewModel.getAlgorithmOptions().observe(this, this::updateAlgorithmSpinner);
        presentationViewModel.getAlgorithmDetail().observe(this, this::showAlgorithmInfo);
        presentationViewModel.getPredictionResult().observe(this, this::onPredictionResult);
        presentationViewModel.getHistoryList().observe(this, logs -> {
            historyAdapter.submitList(logs);
            binding.tvHistoryEmpty.setVisibility(logs == null || logs.isEmpty() ? View.VISIBLE : View.GONE);
        });
        presentationViewModel.getLoading().observe(this, isLoading ->
                binding.progressBar.setVisibility(isLoading != null && isLoading ? View.VISIBLE : View.GONE));
        presentationViewModel.getError().observe(this, errorMessage -> {
            if (!TextUtils.isEmpty(errorMessage)) {
                ToastUtils.showShort(this, errorMessage);
                presentationViewModel.clearError();
            }
        });
        presentationViewModel.getOperationResult().observe(this, result -> {
            if (!TextUtils.isEmpty(result)) {
                ToastUtils.showShort(this, result);
                presentationViewModel.clearOperationResult();
            }
        });
    }

    private void showOriginalImage(FileInfo fileInfo) {
        if (fileInfo == null || fileInfo.getUrl() == null) return;
        binding.ivOriginal.setVisibility(View.VISIBLE);
        String resolved = DehazeSDK.getInstance().resolveUrl(fileInfo.getUrl());
        Glide.with(this).load(resolved)
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(binding.ivOriginal);
        pagerAdapter.setOriginalUrl(resolved);
    }

    private void updateAlgorithmSpinner(List<Option> options) {
        algorithmOptions.clear();
        if (options != null) {
            algorithmOptions.addAll(options);
        }
        ViewUtils.updateAlgorithmSpinner(binding.spinnerAlgorithm, algorithmOptions);
    }

    private void showAlgorithmInfo(Algorithm algorithm) {
        if (algorithm == null) return;
        StringBuilder sb = new StringBuilder();
        sb.append("算法：").append(algorithm.getName() == null ? "-" : algorithm.getName());
        if (algorithm.getDescription() != null && !algorithm.getDescription().isEmpty()) {
            sb.append("（").append(algorithm.getDescription()).append("）");
        }
        if (algorithm.getFlops() != null && !algorithm.getFlops().isEmpty()) {
            sb.append("\nFLOPs：").append(algorithm.getFlops());
        }
        if (algorithm.getSize() != null && !algorithm.getSize().isEmpty()) {
            sb.append("\n模型大小：").append(algorithm.getSize());
        }
        binding.tvAlgorithmInfo.setText(sb.toString());
    }

    private void onPredictionResult(PredResult result) {
        if (result == null) return;
        binding.cardResult.setVisibility(View.VISIBLE);
        pagerAdapter.setResultUrl(DehazeSDK.getInstance().resolveUrl(result.getResultUrl()));
        Long algorithmId = getCurrentAlgorithmId();
        if (algorithmId != null) {
            presentationViewModel.getAlgorithmDetail(algorithmId);
        }
        String info = "耗时：" + (result.getTime() == null ? "-" : result.getTime() + "ms")
                + (Boolean.TRUE.equals(result.getFromCache()) ? "（命中缓存）" : "");
        binding.tvAlgorithmInfo.setText(info);
    }

    private void loadData() {
        presentationViewModel.loadAlgorithmOptions();
        presentationViewModel.loadHistory();
    }

    private void uploadImage(Uri uri) {
        File tempFile = copyToCache(uri);
        if (tempFile != null) presentationViewModel.uploadImage(tempFile);
    }

    private File copyToCache(Uri uri) {
        File tempFile = UriUtils.copyToCache(this, uri);
        if (tempFile == null) {
            ToastUtils.showShort(this, "无法读取所选图片");
        }
        return tempFile;
    }

    /**
     * 处理结果 ViewPager 适配器，显示原图/去雾结果两个页面。
     */
    private static class ResultPagerAdapter extends FragmentStateAdapter {

        private String originalUrl;
        private String resultUrl;
        private final ImagePageFragment[] fragments = new ImagePageFragment[2];

        public ResultPagerAdapter(@NonNull FragmentActivity fragmentActivity) {
            super(fragmentActivity);
        }

        public void setOriginalUrl(String url) {
            this.originalUrl = url;
            ImagePageFragment fragment = fragments[0];
            if (fragment != null) {
                fragment.updateUrl(url);
            } else {
                notifyItemChanged(0);
            }
        }

        public void setResultUrl(String url) {
            this.resultUrl = url;
            ImagePageFragment fragment = fragments[1];
            if (fragment != null) {
                fragment.updateUrl(url);
            } else {
                notifyItemChanged(1);
            }
        }

        @NonNull
        @Override
        public Fragment createFragment(int position) {
            String url = position == 0 ? originalUrl : resultUrl;
            ImagePageFragment fragment = ImagePageFragment.newInstance(url);
            fragments[position] = fragment;
            return fragment;
        }

        @Override
        public int getItemCount() {
            return 2;
        }
    }

    /**
     * 显示单张图片的 Fragment。
     */
    public static class ImagePageFragment extends Fragment {

        private static final String ARG_URL = "image_url";

        private ImageView imageView;
        private String currentUrl;

        public static ImagePageFragment newInstance(String url) {
            ImagePageFragment fragment = new ImagePageFragment();
            Bundle args = new Bundle();
            args.putString(ARG_URL, url);
            fragment.setArguments(args);
            return fragment;
        }

        @Override
        public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container,
                                 Bundle savedInstanceState) {
            View view = inflater.inflate(R.layout.item_image_page, container, false);
            imageView = view.findViewById(R.id.iv_page_image);
            if (currentUrl == null) {
                Bundle args = getArguments();
                currentUrl = args != null ? args.getString(ARG_URL) : null;
            }
            loadImage(currentUrl);
            return view;
        }

        @Override
        public void onDestroyView() {
            super.onDestroyView();
            imageView = null;
        }

        public void updateUrl(String url) {
            currentUrl = url;
            if (imageView != null) {
                loadImage(url);
            }
        }

        private void loadImage(String url) {
            if (url != null && !url.isEmpty()) {
                Glide.with(this).load(DehazeSDK.getInstance().resolveUrl(url))
                        .placeholder(R.drawable.ic_image)
                        .error(R.drawable.ic_broken_image)
                        .into(imageView);
            } else {
                imageView.setImageResource(R.drawable.ic_image);
            }
        }
    }
}
