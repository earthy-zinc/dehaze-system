package com.pei.dehaze.ui.personal;

import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityHelpBinding;
import com.pei.dehaze.ui.common.BaseActivity;

/**
 * 帮助中心 — FAQ 折叠列表（静态）
 */
public class HelpActivity extends BaseActivity {

    private ActivityHelpBinding binding;

    private final String[][] faqs = {
            {"如何使用去雾功能？", "进入「去雾」Tab，上传图像后选择算法并调整参数，点击开始处理即可。"},
            {"支持哪些图像格式？", "支持 JPG、PNG、BMP、TIFF 等常见图像格式。"},
            {"如何查看处理历史？", "在「我的」→「处理历史」中可查看所有历史任务。"},
            {"如何成为会员？", "在「我的」→「我的会员」中查看会员等级和权益，选择套餐开通。"},
            {"处理失败怎么办？", "检查图像格式和大小，确认算法选择正确后重新提交。如仍有问题可提交反馈。"},
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityHelpBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setupActionBar("帮助中心");

        for (int i = 0; i < faqs.length; i++) {
            View item = getLayoutInflater().inflate(R.layout.item_help_faq, binding.faqContainer, false);
            TextView tvQuestion = item.findViewById(R.id.tv_question);
            TextView tvAnswer = item.findViewById(R.id.tv_answer);

            tvQuestion.setText(faqs[i][0]);
            tvAnswer.setText(faqs[i][1]);

            final int index = i;
            item.setOnClickListener(v -> {
                tvAnswer.setVisibility(tvAnswer.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE);
            });

            binding.faqContainer.addView(item);
        }
    }
}
