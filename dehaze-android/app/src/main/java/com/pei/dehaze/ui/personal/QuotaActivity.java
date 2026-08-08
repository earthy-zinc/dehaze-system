package com.pei.dehaze.ui.personal;

import android.os.Bundle;
import android.view.MenuItem;

import androidx.appcompat.app.AppCompatActivity;

import com.pei.dehaze.databinding.ActivityQuotaBinding;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.model.prediction.PredictionQuota;
import com.pei.dehaze.utils.ToastUtils;

/**
 * 我的额度 — 额度明细与进度条
 */
public class QuotaActivity extends AppCompatActivity {

    private ActivityQuotaBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityQuotaBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle("我的额度");
        }

        loadQuota();
    }

    private void loadQuota() {
        ModelAPI.getQuota(RepositoryAdapters.wrap(new com.pei.dehaze.repository.RepositoryCallback<PredictionQuota>() {
            @Override
            public void onSuccess(PredictionQuota quota) {
                runOnUiThread(() -> {
                    if (quota != null) {
                        binding.tvQuotaRemaining.setText(String.valueOf(quota.getRemaining() != null ? quota.getRemaining() : 0));
                        binding.tvQuotaUsed.setText(String.valueOf(quota.getUsed() != null ? quota.getUsed() : 0));
                        binding.tvQuotaTotal.setText(String.valueOf(quota.getTotal() != null ? quota.getTotal() : 0));
                        if (quota.getTotal() != null && quota.getTotal() > 0) {
                            int progress = (int) ((quota.getUsed() != null ? quota.getUsed() : 0) * 100.0 / quota.getTotal());
                            binding.progressQuota.setProgress(Math.min(progress, 100));
                        }
                        binding.tvQuotaHint.setText(quota.getResetDate() != null ? "下次重置: " + quota.getResetDate() : "");
                    }
                });
            }

            @Override
            public void onError(String errorMessage) {
                runOnUiThread(() -> {
                    binding.tvQuotaRemaining.setText("--");
                    binding.tvQuotaUsed.setText("--");
                    binding.tvQuotaTotal.setText("--");
                    binding.progressQuota.setProgress(0);
                    binding.tvQuotaHint.setText("额度数据加载失败");
                });
            }
        }));
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            finish();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }
}
