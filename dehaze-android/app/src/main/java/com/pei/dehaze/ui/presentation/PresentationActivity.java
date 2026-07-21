package com.pei.dehaze.ui.presentation;

import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import androidx.viewpager2.widget.ViewPager2;

import com.bumptech.glide.Glide;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;
import com.pei.dehaze.R;
import com.pei.dehaze.sdk.DehazeSDK;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.ui.common.adapter.PredictionLogAdapter;
import com.pei.dehaze.ui.presentation.viewmodel.PresentationViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;

public class PresentationActivity extends AppCompatActivity {

    private PresentationViewModel presentationViewModel;

    private Toolbar toolbar;
    private ImageView ivOriginal;
    private Spinner spinnerAlgorithm;
    private SeekBar seekBarStrength;
    private TextView tvStrengthValue;
    private MaterialButton btnPredict;
    private ProgressBar progressBar;
    private MaterialCardView cardResult;
    private ViewPager2 viewPager;
    private TabLayout tabLayout;
    private TextView tvAlgorithmInfo;
    private RecyclerView rvHistory;

    private final List<Option> algorithmOptions = new ArrayList<>();
    private PredictionLogAdapter historyAdapter;
    private ResultPagerAdapter pagerAdapter;

    private final ActivityResultLauncher<String> pickImageLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) uploadImage(uri);
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_presentation);

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        toolbar = findViewById(R.id.toolbar);
        ivOriginal = findViewById(R.id.iv_original);
        spinnerAlgorithm = findViewById(R.id.spinner_algorithm);
        seekBarStrength = findViewById(R.id.seek_bar_strength);
        tvStrengthValue = findViewById(R.id.tv_strength_value);
        btnPredict = findViewById(R.id.btn_predict);
        progressBar = findViewById(R.id.progress_bar);
        cardResult = findViewById(R.id.card_result);
        viewPager = findViewById(R.id.view_pager);
        tabLayout = findViewById(R.id.tab_layout);
        tvAlgorithmInfo = findViewById(R.id.tv_algorithm_info);
        rvHistory = findViewById(R.id.rv_history);

        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(v -> finish());

        findViewById(R.id.btn_select_image).setOnClickListener(v ->
                pickImageLauncher.launch("image/*"));

        seekBarStrength.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                tvStrengthValue.setText("强度：" + progress);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        btnPredict.setOnClickListener(v -> onPredictClick());

        pagerAdapter = new ResultPagerAdapter(this);
        viewPager.setAdapter(pagerAdapter);
        new TabLayoutMediator(tabLayout, viewPager,
                (tab, position) -> tab.setText(position == 0 ? "原图" : "去雾结果"))
                .attach();

        historyAdapter = new PredictionLogAdapter();
        rvHistory.setLayoutManager(new LinearLayoutManager(this));
        rvHistory.setAdapter(historyAdapter);
    }

    private void onPredictClick() {
        Long algorithmId = getCurrentAlgorithmId();
        if (algorithmId == null) {
            ToastUtils.showShort(this, "请先选择算法");
            return;
        }
        int strength = seekBarStrength.getProgress();
        presentationViewModel.predict(algorithmId, String.valueOf(strength));
    }

    private Long getCurrentAlgorithmId() {
        int pos = spinnerAlgorithm.getSelectedItemPosition();
        if (pos < 0 || pos >= algorithmOptions.size()) return null;
        Option option = algorithmOptions.get(pos);
        return safeParseLong(option.getValue());
    }

    private void initViewModel() {
        presentationViewModel = new ViewModelProvider(this).get(PresentationViewModel.class);
    }

    private void setupObservers() {
        presentationViewModel.getUploadedFile().observe(this, this::showOriginalImage);
        presentationViewModel.getAlgorithmOptions().observe(this, this::updateAlgorithmSpinner);
        presentationViewModel.getAlgorithmDetail().observe(this, this::showAlgorithmInfo);
        presentationViewModel.getPredictionResult().observe(this, this::onPredictionResult);
        presentationViewModel.getHistoryList().observe(this, logs ->
                historyAdapter.submitList(logs));
        presentationViewModel.getLoading().observe(this, isLoading ->
                progressBar.setVisibility(isLoading != null && isLoading ? View.VISIBLE : View.GONE));
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
        ivOriginal.setVisibility(View.VISIBLE);
        String resolved = DehazeSDK.getInstance().resolveUrl(fileInfo.getUrl());
        Glide.with(this).load(resolved)
                .placeholder(R.drawable.ic_image)
                .error(R.drawable.ic_broken_image)
                .into(ivOriginal);
        pagerAdapter.setOriginalUrl(resolved);
    }

    private void updateAlgorithmSpinner(List<Option> options) {
        algorithmOptions.clear();
        if (options != null) {
            algorithmOptions.addAll(options);
        }
        List<String> labels = new ArrayList<>();
        for (Option opt : algorithmOptions) {
            labels.add(opt.getLabel());
        }
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, labels);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerAlgorithm.setAdapter(adapter);
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
        tvAlgorithmInfo.setText(sb.toString());
    }

    private void onPredictionResult(PredResult result) {
        if (result == null) return;
        cardResult.setVisibility(View.VISIBLE);
        pagerAdapter.setResultUrl(DehazeSDK.getInstance().resolveUrl(result.getResultUrl()));
        Long algorithmId = getCurrentAlgorithmId();
        if (algorithmId != null) {
            presentationViewModel.getAlgorithmDetail(algorithmId.intValue());
        }
        String info = "耗时：" + (result.getTime() == null ? "-" : result.getTime() + "ms")
                + (Boolean.TRUE.equals(result.getFromCache()) ? "（命中缓存）" : "");
        tvAlgorithmInfo.setText(info);
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
        try {
            InputStream is = getContentResolver().openInputStream(uri);
            if (is == null) {
                ToastUtils.showShort(this, "无法读取所选图片");
                return null;
            }
            String fileName = getFileNameFromUri(uri);
            File tempFile = new File(getCacheDir(), fileName != null ? fileName : "upload_temp");
            try (FileOutputStream fos = new FileOutputStream(tempFile)) {
                byte[] buffer = new byte[4096];
                int len;
                while ((len = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, len);
                }
            }
            is.close();
            return tempFile;
        } catch (Exception e) {
            ToastUtils.showShort(this, "读取图片失败: " + e.getMessage());
            return null;
        }
    }

    private String getFileNameFromUri(Uri uri) {
        String result = null;
        if ("content".equals(uri.getScheme())) {
            try (android.database.Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
                if (cursor != null && cursor.moveToFirst()) {
                    int idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME);
                    if (idx >= 0) {
                        result = cursor.getString(idx);
                    }
                }
            }
        }
        if (result == null) {
            result = uri.getLastPathSegment();
        }
        return result;
    }

    private Long safeParseLong(String value) {
        if (value == null) return null;
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /**
     * 处理结果 ViewPager 适配器，显示原图/去雾结果两个页面。
     */
    private static class ResultPagerAdapter extends FragmentStateAdapter {

        private String originalUrl;
        private String resultUrl;

        public ResultPagerAdapter(@NonNull FragmentActivity fragmentActivity) {
            super(fragmentActivity);
        }

        public void setOriginalUrl(String url) {
            this.originalUrl = url;
            notifyItemChanged(0);
        }

        public void setResultUrl(String url) {
            this.resultUrl = url;
            notifyItemChanged(1);
        }

        @NonNull
        @Override
        public Fragment createFragment(int position) {
            String url = position == 0 ? originalUrl : resultUrl;
            return ImagePageFragment.newInstance(url);
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
            ImageView imageView = view.findViewById(R.id.iv_page_image);
            Bundle args = getArguments();
            String url = args != null ? args.getString(ARG_URL) : null;
            if (url != null && !url.isEmpty()) {
                Glide.with(this).load(DehazeSDK.getInstance().resolveUrl(url))
                        .placeholder(R.drawable.ic_image)
                        .error(R.drawable.ic_broken_image)
                        .into(imageView);
            } else {
                imageView.setImageResource(R.drawable.ic_image);
            }
            return view;
        }
    }
}
