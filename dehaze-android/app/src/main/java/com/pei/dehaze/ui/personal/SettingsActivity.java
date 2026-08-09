package com.pei.dehaze.ui.personal;

import android.content.Intent;
import android.os.Bundle;

import androidx.appcompat.app.AlertDialog;
import androidx.lifecycle.ViewModelProvider;

import com.pei.dehaze.MainActivity;
import com.pei.dehaze.databinding.ActivitySettingsBinding;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.AuthAPI;
import com.pei.dehaze.ui.common.BaseActivity;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.ui.notify.NotifyActivity;
import com.pei.dehaze.utils.ToastUtils;

import java.io.File;

/**
 * 系统设置 — 缓存清理 / 通知入口 / 版本号 / 退出登录
 */
public class SettingsActivity extends BaseActivity {

    private ActivitySettingsBinding binding;
    private SettingsViewModel viewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivitySettingsBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setupActionBar("系统设置");

        binding.tvVersion.setText("v1.0");

        viewModel = new ViewModelProvider(this).get(SettingsViewModel.class);
        observeError(viewModel);
        observeOperationResult(viewModel, this::navigateToLogin);

        binding.itemClearCache.setOnClickListener(v -> {
            try {
                File cacheDir = getCacheDir();
                deleteDir(cacheDir);
                ToastUtils.showShort(this, "缓存已清理");
            } catch (Exception e) {
                ToastUtils.showShort(this, "缓存清理失败");
            }
        });

        binding.itemNotify.setOnClickListener(v ->
                startActivity(new Intent(this, NotifyActivity.class)));

        binding.itemLogout.setOnClickListener(v ->
                new AlertDialog.Builder(this)
                        .setTitle("退出登录")
                        .setMessage("确定要退出当前账号吗？")
                        .setPositiveButton("确定", (d, w) -> viewModel.logout())
                        .setNegativeButton("取消", null)
                        .show());
    }

    private void navigateToLogin() {
        Intent intent = new Intent(this, MainActivity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
        intent.putExtra("logout", true);
        startActivity(intent);
        finishAffinity();
    }

    private boolean deleteDir(File dir) {
        if (dir != null && dir.isDirectory()) {
            String[] children = dir.list();
            if (children != null) {
                for (String child : children) {
                    if (!deleteDir(new File(dir, child))) {
                        return false;
                    }
                }
            }
        }
        return dir != null && dir.delete();
    }

    public static class SettingsViewModel extends BaseViewModel {
        public void logout() {
            AuthAPI.logout(RepositoryAdapters.wrap(withLoading(v -> operationResult.postValue("已退出登录"))));
        }
    }
}
