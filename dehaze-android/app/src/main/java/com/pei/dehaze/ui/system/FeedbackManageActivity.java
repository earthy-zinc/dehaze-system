package com.pei.dehaze.ui.system;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import com.pei.dehaze.ui.common.BaseActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityManageListBinding;
import com.pei.dehaze.sdk.model.feedback.FeedbackPageVO;
import com.pei.dehaze.ui.system.viewmodel.FeedbackManageViewModel;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;

import java.util.List;

/**
 * 反馈评价管理（sys:feedback:*）— 列表查看、回复、关闭
 */
public class FeedbackManageActivity extends BaseActivity {

    private FeedbackManageViewModel viewModel;
    private ActivityManageListBinding binding;
    private FeedbackManageAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityManageListBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        setSupportActionBar(binding.toolbar);
        binding.toolbar.setTitle("反馈评价管理");
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        ArrayAdapter<String> statusAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item,
                new String[]{"全部", "待处理", "已回复", "已关闭"});
        statusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        binding.spinnerStatus.setAdapter(statusAdapter);

        adapter = new FeedbackManageAdapter();
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(adapter);

        binding.swipeRefresh.setOnRefreshListener(this::loadData);

        binding.btnSearch.setOnClickListener(v -> {
            viewModel.setKeywords(binding.etKeywords.getText().toString().trim());
            loadData();
        });

        binding.btnReset.setOnClickListener(v -> {
            binding.etKeywords.setText("");
            binding.spinnerStatus.setSelection(0);
            viewModel.resetQuery();
            loadData();
        });

        binding.btnPrev.setOnClickListener(v -> viewModel.prevPage());
        binding.btnNext.setOnClickListener(v -> viewModel.nextPage());
    }

    private void initViewModel() {
        viewModel = new ViewModelProvider(this).get(FeedbackManageViewModel.class);
    }

    private void setupObservers() {
        viewModel.getItemList().observe(this, items -> {
            adapter.notifyDataSetChanged();
            binding.tvEmpty.setVisibility(items == null || items.isEmpty() ? View.VISIBLE : View.GONE);
            updatePageInfo();
        });

        viewModel.getTotal().observe(this, total -> updatePageInfo());

        viewModel.getLoading().observe(this, isLoading ->
                binding.swipeRefresh.setRefreshing(isLoading != null && isLoading));

        observeError(viewModel);
        observeOperationResult(viewModel, null);
    }

    private void loadData() {
        viewModel.loadData();
    }

    private void updatePageInfo() {
        long total = viewModel.getTotal().getValue() != null ? viewModel.getTotal().getValue() : 0L;
        int pageNum = viewModel.getPageNum();
        int pageSize = viewModel.getPageSize();
        int totalPages = Math.max(1, (int) Math.ceil(total * 1.0 / pageSize));
        binding.tvPageInfo.setText("第 " + pageNum + " 页 / 共 " + totalPages + " 页 (共 " + total + " 条)");
    }

    // ---- 回复对话框 ----
    private void showReplyDialog(FeedbackPageVO item) {
        if (item.getId() == null) return;
        View formView = LayoutInflater.from(this).inflate(R.layout.dialog_feedback_reply, null);
        EditText etReply = formView.findViewById(R.id.et_reply);

        new AlertDialog.Builder(this)
                .setTitle("回复反馈 — " + StringUtils.safe(item.getTitle(), "无标题"))
                .setView(formView)
                .setPositiveButton("发送回复", (dialog, which) -> {
                    String content = etReply.getText().toString().trim();
                    if (TextUtils.isEmpty(content)) {
                        ToastUtils.showShort(this, "请输入回复内容");
                        return;
                    }
                    viewModel.replyFeedback(item.getId(), content);
                })
                .setNegativeButton("取消", null)
                .show();
    }

    // ---- 关闭确认 ----
    private void showCloseDialog(FeedbackPageVO item) {
        if (item.getId() == null) return;
        View formView = LayoutInflater.from(this).inflate(R.layout.dialog_feedback_reply, null);
        EditText etReply = formView.findViewById(R.id.et_reply);
        etReply.setHint("关闭原因（选填）");

        new AlertDialog.Builder(this)
                .setTitle("关闭反馈 — " + StringUtils.safe(item.getTitle(), "无标题"))
                .setView(formView)
                .setPositiveButton("确认关闭", (dialog, which) -> {
                    String reason = etReply.getText().toString().trim();
                    if (TextUtils.isEmpty(reason)) {
                        reason = "管理员关闭";
                    }
                    viewModel.closeFeedback(item.getId(), reason);
                })
                .setNegativeButton("取消", null)
                .show();
    }

    // ---- Adapter ----
    private class FeedbackManageAdapter extends RecyclerView.Adapter<FeedbackManageAdapter.ViewHolder> {

        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_feedback_manage, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            List<?> list = viewModel.getItemList().getValue();
            if (list == null || position >= list.size()) return;
            Object obj = list.get(position);
            if (!(obj instanceof FeedbackPageVO)) return;
            FeedbackPageVO item = (FeedbackPageVO) obj;

            holder.tvTitle.setText(StringUtils.safe(item.getTitle(), "无标题"));

            String status = StringUtils.safe(item.getStatus(), "--");
            holder.tvStatus.setText(status);
            int statusColor;
            switch (status) {
                case "PENDING": statusColor = 0xFFFF9800; break;
                case "REPLIED": statusColor = 0xFF4CAF50; break;
                case "CLOSED": statusColor = 0xFF757575; break;
                default: statusColor = 0xFF757575; break;
            }
            holder.tvStatus.setTextColor(statusColor);

            holder.tvType.setText("类型: " + StringUtils.safe(item.getFeedbackType(), "--"));
            holder.tvUser.setText("用户: " + StringUtils.safe(item.getUsername(), "--"));

            String content = StringUtils.safe(item.getContent());
            holder.tvContent.setText(content);
            holder.tvContent.setVisibility(TextUtils.isEmpty(content) ? View.GONE : View.VISIBLE);

            holder.tvTime.setText(StringUtils.safe(item.getCreateTime(), ""));

            int priority = item.getPriority() != null ? item.getPriority() : 0;
            String priorityText;
            int priorityColor;
            if (priority >= 3) {
                priorityText = "高优先级";
                priorityColor = 0xFFF44336;
            } else if (priority == 2) {
                priorityText = "中优先级";
                priorityColor = 0xFFFF9800;
            } else {
                priorityText = "低优先级";
                priorityColor = 0xFF4CAF50;
            }
            holder.tvPriority.setText(priorityText);
            holder.tvPriority.setTextColor(priorityColor);

            holder.tvReply.setOnClickListener(v -> showReplyDialog(item));
            holder.tvClose.setOnClickListener(v -> showCloseDialog(item));
            holder.tvDelete.setOnClickListener(v -> showCloseDialog(item)); // SDK 无 delete，用 close 代替
        }

        @Override
        public int getItemCount() {
            List<?> list = viewModel.getItemList().getValue();
            return list != null ? list.size() : 0;
        }

        class ViewHolder extends RecyclerView.ViewHolder {
            TextView tvTitle, tvStatus, tvType, tvUser, tvContent, tvTime, tvPriority;
            TextView tvReply, tvClose, tvDelete;

            ViewHolder(View itemView) {
                super(itemView);
                tvTitle = itemView.findViewById(R.id.tv_title);
                tvStatus = itemView.findViewById(R.id.tv_status);
                tvType = itemView.findViewById(R.id.tv_type);
                tvUser = itemView.findViewById(R.id.tv_user);
                tvContent = itemView.findViewById(R.id.tv_content);
                tvTime = itemView.findViewById(R.id.tv_time);
                tvPriority = itemView.findViewById(R.id.tv_priority);
                tvReply = itemView.findViewById(R.id.tv_reply);
                tvClose = itemView.findViewById(R.id.tv_close);
                tvDelete = itemView.findViewById(R.id.tv_delete);
            }
        }
    }
}
