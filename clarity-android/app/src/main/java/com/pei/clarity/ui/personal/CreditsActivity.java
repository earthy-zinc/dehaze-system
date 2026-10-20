package com.pei.clarity.ui.personal;

import android.os.Bundle;
import android.view.View;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModelProvider;

import com.pei.clarity.databinding.ActivityCreditsBinding;
import com.pei.clarity.repository.RepositoryAdapters;
import com.pei.clarity.sdk.api.AiBillingAPI;
import com.pei.clarity.sdk.model.billing.BalanceVO;
import com.pei.clarity.ui.common.BaseActivity;
import com.pei.clarity.ui.common.BaseViewModel;

/**
 * 我的积分余额 —— 余额与今日/本月已用量
 */
public class CreditsActivity extends BaseActivity {

    private ActivityCreditsBinding binding;
    private CreditsViewModel viewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityCreditsBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setupActionBar("我的积分余额");

        viewModel = new ViewModelProvider(this).get(CreditsViewModel.class);

        viewModel.getBalance().observe(this, balance -> {
            if (balance == null) return;
            binding.tvCreditsBalance.setText(formatCredits(balance.getCreditsBalance()));
            binding.tvArrearsHint.setVisibility(
                    Boolean.TRUE.equals(balance.getArrearsStatus()) ? View.VISIBLE : View.GONE);
            binding.tvDailyUsage.setText(formatUsage(balance.getDailyUsed(), balance.getDailyLimit()));
            binding.tvMonthlyUsage.setText(formatUsage(balance.getMonthlyUsed(), balance.getMonthlyLimit()));
        });
        observeError(viewModel);

        binding.tvCreditsBalance.setText("--");
        binding.tvDailyUsage.setText("--");
        binding.tvMonthlyUsage.setText("--");
        binding.tvCreditsHint.setText("积分用于图像处理等 AI 能力消耗，额度按自然月重置");

        viewModel.loadBalance();
    }

    private String formatCredits(String credits) {
        return credits == null ? "--" : credits;
    }

    private String formatUsage(Integer used, Integer limit) {
        if (used == null) return "--";
        return limit == null ? String.valueOf(used) : used + " / " + limit;
    }

    public static class CreditsViewModel extends BaseViewModel {
        private final MutableLiveData<BalanceVO> balance = new MutableLiveData<>();

        public LiveData<BalanceVO> getBalance() {
            return balance;
        }

        public void loadBalance() {
            AiBillingAPI.getBalance(RepositoryAdapters.wrap(withLoading(data -> balance.postValue(data))));
        }
    }
}
