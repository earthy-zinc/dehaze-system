package com.pei.clarity.ui.presentation;

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
import com.pei.clarity.ui.common.BaseActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.viewpager2.adapter.FragmentStateAdapter;

import com.bumptech.glide.Glide;
import com.google.android.material.tabs.TabLayoutMediator;
import com.pei.clarity.R;
import com.pei.clarity.databinding.ActivityPresentationBinding;
import com.pei.clarity.sdk.model.ai_model.AiModelVO;
import com.pei.clarity.sdk.model.file.FileInfo;
import com.pei.clarity.sdk.model.image_task.DehazeParams;
import com.pei.clarity.sdk.model.image_task.ImageTaskResponse;
import com.pei.clarity.ui.common.adapter.ImageTaskAdapter;
import com.pei.clarity.ui.presentation.viewmodel.PresentationViewModel;
import com.pei.clarity.utils.StatePlaceholder;
import com.pei.clarity.utils.ToastUtils;
import com.pei.clarity.utils.UriUtils;
import com.pei.clarity.utils.ViewUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;

public class PresentationActivity extends BaseActivity {

    private PresentationViewModel presentationViewModel;
    private ActivityPresentationBinding binding;
    private StatePlaceholder statePlaceholder;

    private final List<AiModelVO> modelOptions = new ArrayList<>();
    private ImageTaskAdapter historyAdapter;
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

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        setupToolbar(binding.toolbar, null);

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

        statePlaceholder = new StatePlaceholder(binding.statePlaceholder.getRoot());
        statePlaceholder.showEmpty("请上传图片并点击\"开始去雾处理\"", R.drawable.ic_image);

        historyAdapter = new ImageTaskAdapter();
        binding.rvHistory.setLayoutManager(new LinearLayoutManager(this));
        binding.rvHistory.setAdapter(historyAdapter);
    }

    private void onPredictClick() {
        String modelId = getCurrentModelId();
        if (modelId == null) {
            ToastUtils.showShort(this, "请先选择模型");
            return;
        }
        int strength = binding.seekBarStrength.getProgress();
        DehazeParams params = new DehazeParams(strength, 100, 100, 30);
        new AlertDialog.Builder(this)
                .setTitle("确认处理")
                .setMessage("确认开始去雾处理？")
                .setPositiveButton("确定", (d, w) ->
                        presentationViewModel.predict(modelId, params))
                .setNegativeButton("取消", null)
                .show();
    }

    private String getCurrentModelId() {
        int pos = binding.spinnerModel.getSelectedItemPosition();
        if (pos < 0 || pos >= modelOptions.size()) return null;
        return modelOptions.get(pos).getModelId();
    }

    private String getCurrentModelName() {
        int pos = binding.spinnerModel.getSelectedItemPosition();
        if (pos < 0 || pos >= modelOptions.size()) return null;
        AiModelVO model = modelOptions.get(pos);
        return model.getDisplayName() != null ? model.getDisplayName() : model.getModelId();
    }

    private void initViewModel() {
        presentationViewModel = new ViewModelProvider(this).get(PresentationViewModel.class);
    }

    private void setupObservers() {
        presentationViewModel.getUploadedFile().observe(this, this::showOriginalImage);
        presentationViewModel.getModels().observe(this, this::updateModelSpinner);
        presentationViewModel.getPredictionResult().observe(this, this::onPredictionResult);
        presentationViewModel.getHistoryList().observe(this, logs -> {
            historyAdapter.submitList(logs);
            binding.tvHistoryEmpty.setVisibility(logs == null || logs.isEmpty() ? View.VISIBLE : View.GONE);
        });
        presentationViewModel.getLoading().observe(this, isLoading -> {
            boolean loading = isLoading != null && isLoading;
            binding.progressBar.setVisibility(loading ? View.VISIBLE : View.GONE);
            if (loading) {
                statePlaceholder.showLoading("正在处理中…");
            }
        });
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
        String resolved = fileInfo.getUrl();
        Glide.with(this).load(resolved)
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(binding.ivOriginal);
        pagerAdapter.setOriginalUrl(resolved);
        statePlaceholder.hide();
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
        pagerAdapter.setResultUrl(result.getResultUrl());
        String modelName = getCurrentModelName();
        String info = (modelName == null ? "" : "模型：" + modelName + "\n")
                + "耗时：" + (result.getTime() == null ? "-" : result.getTime() + "s");
        binding.tvModelInfo.setText(info);
    }

    private void loadData() {
        presentationViewModel.loadModels();
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
                Glide.with(this).load(url)
                        .placeholder(R.drawable.ic_image)
                        .error(R.drawable.ic_broken_image)
                        .into(imageView);
            } else {
                imageView.setImageResource(R.drawable.ic_image);
            }
        }
    }
}
