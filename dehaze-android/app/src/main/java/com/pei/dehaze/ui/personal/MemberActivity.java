package com.pei.dehaze.ui.personal;

import android.os.Bundle;
import android.view.MenuItem;

import androidx.appcompat.app.AppCompatActivity;

import com.pei.dehaze.databinding.ActivityMemberBinding;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.MemberAPI;
import com.pei.dehaze.utils.ToastUtils;

/**
 * 我的会员 — 等级/成长值/权益
 */
public class MemberActivity extends AppCompatActivity {

    private ActivityMemberBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMemberBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle("我的会员");
        }

        loadProfile();
    }

    private void loadProfile() {
        MemberAPI.getProfile(RepositoryAdapters.wrap(new com.pei.dehaze.repository.RepositoryCallback<com.pei.dehaze.sdk.model.member.MemberProfileVO>() {
            @Override
            public void onSuccess(com.pei.dehaze.sdk.model.member.MemberProfileVO profile) {
                runOnUiThread(() -> {
                    if (profile != null) {
                        binding.tvMemberLevel.setText(profile.getLevelName() != null ? profile.getLevelName() : "普通用户");
                        binding.tvGrowthValue.setText("成长值: " + (profile.getGrowthValue() != null ? profile.getGrowthValue() : 0));
                        String desc = "月度去雾: " + (profile.getMonthlyDehazeQuota() != null ? profile.getMonthlyDehazeQuota() : 0)
                                + " / 月度评估: " + (profile.getMonthlyEvaluateQuota() != null ? profile.getMonthlyEvaluateQuota() : 0);
                        binding.tvMemberDesc.setText(desc);
                    }
                });
            }

            @Override
            public void onError(String errorMessage) {
                runOnUiThread(() -> {
                    binding.tvMemberLevel.setText("普通用户");
                    binding.tvGrowthValue.setText("成长值: --");
                    binding.tvMemberDesc.setText("会员信息加载失败");
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
