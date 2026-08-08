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
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivityManageListBinding;
import com.pei.dehaze.sdk.model.order.OrderPageVO;
import com.pei.dehaze.ui.system.viewmodel.OrderManageViewModel;
import com.pei.dehaze.utils.StringUtils;
import com.pei.dehaze.utils.ToastUtils;

import java.util.List;

/**
 * 订单管理（sys:order:*）— 列表查看、取消订单、退款处理
 */
public class OrderManageActivity extends AppCompatActivity {

    private OrderManageViewModel viewModel;
    private ActivityManageListBinding binding;
    private OrderManageAdapter adapter;

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
        binding.toolbar.setTitle("订单管理");
        binding.toolbar.setNavigationOnClickListener(v -> finish());

        ArrayAdapter<String> statusAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item,
                new String[]{"全部", "待支付", "已支付", "已取消", "已退款"});
        statusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        binding.spinnerStatus.setAdapter(statusAdapter);

        adapter = new OrderManageAdapter();
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
        viewModel = new ViewModelProvider(this).get(OrderManageViewModel.class);
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

        viewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(this, errorMessage);
                viewModel.clearError();
            }
        });

        viewModel.getOperationResult().observe(this, result -> {
            if (result != null && !result.isEmpty()) {
                ToastUtils.showShort(this, result);
                viewModel.clearOperationResult();
            }
        });
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

    // ---- 退款处理对话框 ----
    private void showRefundDialog(OrderPageVO item) {
        if (item.getOrderNo() == null) return;
        View formView = LayoutInflater.from(this).inflate(R.layout.dialog_refund_form, null);
        EditText etReason = formView.findViewById(R.id.et_reason);

        new AlertDialog.Builder(this)
                .setTitle("退款处理 — " + item.getOrderNo())
                .setView(formView)
                .setPositiveButton("确认退款", (dialog, which) -> {
                    String reason = etReason.getText().toString().trim();
                    if (TextUtils.isEmpty(reason)) {
                        ToastUtils.showShort(this, "请输入退款原因");
                        return;
                    }
                    viewModel.applyRefund(item.getOrderNo(), reason);
                })
                .setNegativeButton("取消", null)
                .show();
    }

    // ---- 取消订单确认 ----
    private void showCancelOrderDialog(OrderPageVO item) {
        if (item.getOrderNo() == null) return;
        new AlertDialog.Builder(this)
                .setTitle("取消订单确认")
                .setMessage("确认取消订单「" + item.getOrderNo() + "」吗？")
                .setPositiveButton("确定", (dialog, which) ->
                        viewModel.cancelOrder(item.getOrderNo(), "管理员手动取消"))
                .setNegativeButton("取消", null)
                .show();
    }

    // ---- Adapter ----
    private class OrderManageAdapter extends RecyclerView.Adapter<OrderManageAdapter.ViewHolder> {

        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_order_manage, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            List<?> list = viewModel.getItemList().getValue();
            if (list == null || position >= list.size()) return;
            Object obj = list.get(position);
            if (!(obj instanceof OrderPageVO)) return;
            OrderPageVO item = (OrderPageVO) obj;

            holder.tvOrderNo.setText("订单: " + StringUtils.safe(item.getOrderNo(), "--"));

            String status = StringUtils.safe(item.getStatus(), "--");
            holder.tvStatus.setText(status);
            // 根据状态设置颜色
            int statusColor;
            switch (status) {
                case "PAID": statusColor = 0xFF4CAF50; break;
                case "PENDING": statusColor = 0xFFFF9800; break;
                case "CANCELLED": statusColor = 0xFFF44336; break;
                case "REFUNDED": statusColor = 0xFF9C27B0; break;
                default: statusColor = 0xFF757575; break;
            }
            holder.tvStatus.setTextColor(statusColor);

            holder.tvPackage.setText(StringUtils.safe(item.getPackageName(), "--"));

            if (item.getPaidAmount() != null) {
                holder.tvAmount.setText("¥" + String.format("%.2f", item.getPaidAmount()));
            } else {
                holder.tvAmount.setText("--");
            }

            holder.tvUser.setText("用户: " + StringUtils.safe(item.getUsername(), "--"));
            holder.tvTime.setText(StringUtils.safe(item.getCreateTime(), ""));

            holder.tvRefund.setOnClickListener(v -> showRefundDialog(item));
            holder.tvCancel.setOnClickListener(v -> showCancelOrderDialog(item));
        }

        @Override
        public int getItemCount() {
            List<?> list = viewModel.getItemList().getValue();
            return list != null ? list.size() : 0;
        }

        class ViewHolder extends RecyclerView.ViewHolder {
            TextView tvOrderNo, tvStatus, tvPackage, tvAmount, tvUser, tvTime;
            TextView tvRefund, tvCancel;

            ViewHolder(View itemView) {
                super(itemView);
                tvOrderNo = itemView.findViewById(R.id.tv_order_no);
                tvStatus = itemView.findViewById(R.id.tv_status);
                tvPackage = itemView.findViewById(R.id.tv_package);
                tvAmount = itemView.findViewById(R.id.tv_amount);
                tvUser = itemView.findViewById(R.id.tv_user);
                tvTime = itemView.findViewById(R.id.tv_time);
                tvRefund = itemView.findViewById(R.id.tv_refund);
                tvCancel = itemView.findViewById(R.id.tv_cancel);
            }
        }
    }
}
