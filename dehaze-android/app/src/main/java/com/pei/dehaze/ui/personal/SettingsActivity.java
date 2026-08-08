package com.pei.dehaze.ui.personal;

import android.content.Intent;
import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.widget.TextView;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.pei.dehaze.MainActivity;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivitySettingsBinding;
import com.pei.dehaze.sdk.api.AuthAPI;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.ui.notify.NotifyActivity;
import com.pei.dehaze.utils.ToastUtils;

import java.io.File;

/**
 * 系统设置 — 缓存清理 / 通知入口 / 版本号 / 退出登录
 */
public class SettingsActivity extends AppCompatActivity {

    private ActivitySettingsBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivitySettingsBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle("系统设置");
        }

        binding.tvVersion.setText("v1.0");

        binding.itemClearCache.setOnClickListener(v -> {
            // 清理应用缓存目录
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
                        .setPositiveButton("确定", (d, w) -> {
                            AuthAPI.logout(RepositoryAdapters.wrap(new com.pei.dehaze.repository.RepositoryCallback<Void>() {
                                @Override
                                public void onSuccess(Void data) {
                                    runOnUiThread(() -> navigateToLogin());
                                }

                                @Override
                                public void onError(String errorMessage) {
                                    runOnUiThread(() -> ToastUtils.showShort(SettingsActivity.this, errorMessage));
                                }
                            }));
                        })
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

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            finish();
            return true;
        }
        return super.onOptionsItemSelected(item);
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
}
