package com.pei.dehaze.ui.messages.detail;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import com.pei.dehaze.databinding.ActivityMessagesDetailBinding;

public class MessagesDetailActivity extends AppCompatActivity {

    private ActivityMessagesDetailBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMessagesDetailBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        String title = getIntent().getStringExtra("message_title");
        String type = getIntent().getStringExtra("message_type");
        String content = getIntent().getStringExtra("message_content");
        String time = getIntent().getStringExtra("message_time");

        binding.tvTitle.setText(title != null ? title : "消息详情");
        binding.tvType.setText(type != null ? type : "通知");
        binding.tvContent.setText(content != null ? content : "");
        binding.tvTime.setText(time != null ? time : "");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        binding = null;
    }
}
