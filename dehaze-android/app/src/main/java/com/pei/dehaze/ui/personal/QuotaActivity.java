package com.pei.dehaze.ui.personal;

import android.os.Bundle;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModelProvider;

import com.pei.dehaze.databinding.ActivityQuotaBinding;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.model.prediction.PredictionQuota;
import com.pei.dehaze.ui.common.BaseActivity;
import com.pei.dehaze.ui.common.BaseViewModel;

/**
 * 我的额度 — 额度明细与进度条
 */
public class QuotaActivity extends BaseActivity {

    private ActivityQuotaBinding binding;
    private QuotaViewModel viewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityQuotaBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setupActionBar("我的额度");

        viewModel = new ViewModelProvider(this).get(QuotaViewModel.class);

        viewModel.getQuota().observe(this, quota -> {
            if (quota == null) return;
            binding.tvQuotaRemaining.setText(String.valueOf(quota.getRemaining() != null ? quota.getRemaining() : 0));
            binding.tvQuotaUsed.setText(String.valueOf(quota.getUsed() != null ? quota.getUsed() : 0));
            binding.tvQuotaTotal.setText(String.valueOf(quota.getTotal() != null ? quota.getTotal() : 0));
            if (quota.getTotal() != null && quota.getTotal() > 0) {
                int progress = (int) ((quota.getUsed() != null ? quota.getUsed() : 0) * 100.0 / quota.getTotal());
                binding.progressQuota.setProgress(Math.min(progress, 100));
            }
            binding.tvQuotaHint.setText(quota.getResetDate() != null ? "下次重置: " + quota.getResetDate() : "");
        });
        observeError(viewModel);

        // 加载前/失败时保留占位符
        binding.tvQuotaRemaining.setText("--");
        binding.tvQuotaUsed.setText("--");
        binding.tvQuotaTotal.setText("--");

        viewModel.loadQuota();
    }

    public static class QuotaViewModel extends BaseViewModel {
        private final MutableLiveData<PredictionQuota> quota = new MutableLiveData<>();

        public LiveData<PredictionQuota> getQuota() {
            return quota;
        }

        public void loadQuota() {
            ModelAPI.getQuota(RepositoryAdapters.wrap(withLoading(data -> quota.postValue(data))));
        }
    }
}
