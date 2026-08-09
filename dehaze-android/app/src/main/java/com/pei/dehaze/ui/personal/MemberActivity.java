package com.pei.dehaze.ui.personal;

import android.os.Bundle;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModelProvider;

import com.pei.dehaze.databinding.ActivityMemberBinding;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.MemberAPI;
import com.pei.dehaze.sdk.model.member.MemberProfileVO;
import com.pei.dehaze.ui.common.BaseActivity;
import com.pei.dehaze.ui.common.BaseViewModel;

/**
 * 我的会员 — 等级/成长值/权益
 */
public class MemberActivity extends BaseActivity {

    private ActivityMemberBinding binding;
    private MemberViewModel viewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMemberBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setupActionBar("我的会员");

        viewModel = new ViewModelProvider(this).get(MemberViewModel.class);

        viewModel.getProfile().observe(this, profile -> {
            if (profile == null) return;
            binding.tvMemberLevel.setText(profile.getLevelName() != null ? profile.getLevelName() : "普通用户");
            binding.tvGrowthValue.setText("成长值: " + (profile.getGrowthValue() != null ? profile.getGrowthValue() : 0));
            String desc = "月度去雾: " + (profile.getMonthlyDehazeQuota() != null ? profile.getMonthlyDehazeQuota() : 0)
                    + " / 月度评估: " + (profile.getMonthlyEvaluateQuota() != null ? profile.getMonthlyEvaluateQuota() : 0);
            binding.tvMemberDesc.setText(desc);
        });
        observeError(viewModel);

        viewModel.loadProfile();
    }

    public static class MemberViewModel extends BaseViewModel {
        private final MutableLiveData<MemberProfileVO> profile = new MutableLiveData<>();

        public LiveData<MemberProfileVO> getProfile() {
            return profile;
        }

        public void loadProfile() {
            MemberAPI.getProfile(RepositoryAdapters.wrap(withLoading(data -> profile.postValue(data))));
        }
    }
}
