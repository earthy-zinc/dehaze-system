package com.pei.dehaze.ui.personal;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityAboutBinding;

/**
 * 关于我们 — Logo + 版本号 + 简介 + 隐私政策/用户协议
 */
public class AboutActivity extends AppCompatActivity {

    private ActivityAboutBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityAboutBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle("关于我们");
        }

        binding.tvAppName.setText("图像去雾系统");
        binding.tvVersion.setText("v1.0");
        binding.tvDescription.setText("基于深度学习的图像去雾处理平台，提供高效的图像去雾、对比评估等智能图像处理能力。");

        binding.itemPrivacy.setOnClickListener(v -> openUrl("https://example.com/privacy"));
        binding.itemTerms.setOnClickListener(v -> openUrl("https://example.com/terms"));
    }

    private void openUrl(String url) {
        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
        startActivity(intent);
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
