package com.pei.dehaze.ui.personal;

import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivitySimpleListBinding;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.OrderAPI;
import com.pei.dehaze.sdk.model.order.MyOrderQuery;
import com.pei.dehaze.sdk.model.order.MyOrderVO;
import com.pei.dehaze.ui.common.BaseActivity;
import com.pei.dehaze.ui.common.BaseLoadMoreViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * 我的订单 — 卡片化列表（订单号 + 套餐名 + 金额 + 状态徽章 + 下单时间 + 操作按钮）
 *
 * <p>分页状态与请求由 {@link OrderListViewModel} 持有，Activity 仅负责展示与交互。
 */
public class OrdersActivity extends BaseActivity {

    private ActivitySimpleListBinding binding;
    private OrderListViewModel viewModel;
    private OrderAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivitySimpleListBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setupActionBar("我的订单");

        viewModel = new ViewModelProvider(this).get(OrderListViewModel.class);
        adapter = new OrderAdapter(new OrderAdapter.OnOrderActionListener() {
            @Override
            public void onPay(MyOrderVO order) {
                ToastUtils.showShort(OrdersActivity.this, "支付功能开发中");
            }

            @Override
            public void onCancel(MyOrderVO order) {
                showCancelDialog(order);
            }

            @Override
            public void onDetail(MyOrderVO order) {
                ToastUtils.showShort(OrdersActivity.this, "订单详情: " + order.getOrderNo());
            }
        });
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(adapter);

        binding.recyclerView.addOnScrollListener(new RecyclerView.OnScrollListener() {
            @Override
            public void onScrolled(@NonNull RecyclerView recyclerView, int dx, int dy) {
                LinearLayoutManager lm = (LinearLayoutManager) recyclerView.getLayoutManager();
                if (lm != null && lm.findLastVisibleItemPosition() + 1 >= adapter.getItemCount()
                        && !Boolean.TRUE.equals(viewModel.getLoading().getValue())) {
                    viewModel.loadMore();
                }
            }
        });

        binding.swipeRefresh.setOnRefreshListener(() -> viewModel.reload());

        viewModel.getOrders().observe(this, list -> {
            adapter.submitList(list);
            if (list == null || list.isEmpty()) {
                binding.emptyView.setVisibility(View.VISIBLE);
                binding.emptyText.setText("暂无订单");
            } else {
                binding.emptyView.setVisibility(View.GONE);
            }
        });
        viewModel.getLoading().observe(this, loading ->
                binding.swipeRefresh.setRefreshing(Boolean.TRUE.equals(loading)));

        observeError(viewModel);
        observeOperationResult(viewModel, () -> viewModel.reload());

        viewModel.reload();
    }

    private void showCancelDialog(MyOrderVO order) {
        new AlertDialog.Builder(this)
                .setTitle("取消订单")
                .setMessage("确定取消订单「" + (order.getOrderNo() != null ? order.getOrderNo() : "") + "」吗？")
                .setPositiveButton("确定", (dialog, which) -> {
                    if (order.getOrderNo() != null) {
                        viewModel.cancelOrder(order.getOrderNo());
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    // region ViewModel

    public static class OrderListViewModel extends BaseLoadMoreViewModel<MyOrderVO> {

        public OrderListViewModel() {
            super(20);
        }

        @Override
        protected void loadPage() {
            MyOrderQuery query = new MyOrderQuery();
            query.setPageNum(pageNum);
            query.setPageSize(pageSize);
            OrderAPI.listMy(query, RepositoryAdapters.wrap(withLoading(data ->
                    onPageLoaded(data.getList(), data.getTotal()))));
        }

        public LiveData<List<MyOrderVO>> getOrders() {
            return itemList;
        }

        public void cancelOrder(String orderNo) {
            OrderAPI.cancel(orderNo, null, RepositoryAdapters.wrap(withLoading(v ->
                    operationResult.postValue("订单已取消"))));
        }
    }

    // endregion

    // region Adapter

    static class OrderAdapter extends RecyclerView.Adapter<OrderAdapter.VH> {
        private final List<MyOrderVO> items = new ArrayList<>();
        private final OnOrderActionListener actionListener;

        interface OnOrderActionListener {
            void onPay(MyOrderVO order);
            void onCancel(MyOrderVO order);
            void onDetail(MyOrderVO order);
        }

        OrderAdapter(OnOrderActionListener actionListener) {
            this.actionListener = actionListener;
        }

        void submitList(List<MyOrderVO> newItems) {
            items.clear();
            if (newItems != null) items.addAll(newItems);
            notifyDataSetChanged();
        }

        @NonNull @Override
        public VH onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View v = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_personal_order, parent, false);
            return new VH(v);
        }

        @Override
        public void onBindViewHolder(@NonNull VH holder, int position) {
            MyOrderVO item = items.get(position);

            holder.tvOrderNo.setText("订单号: " + (item.getOrderNo() != null ? item.getOrderNo() : "--"));
            holder.tvPackageName.setText(item.getPackageName() != null ? item.getPackageName() : "未知套餐");
            holder.tvAmount.setText(formatAmount(item.getPaidAmount()));
            holder.tvTime.setText(item.getCreateTime() != null ? item.getCreateTime() : "");

            // 状态徽章
            String status = item.getStatus() != null ? item.getStatus() : "";
            holder.tvStatus.setText(mapStatusLabel(status));
            holder.tvStatus.setBackground(createStatusBg(mapStatusColor(status)));

            // 操作按钮：根据状态显示/隐藏
            boolean isPending = "待支付".equals(status);
            holder.btnPay.setVisibility(isPending ? View.VISIBLE : View.GONE);
            holder.btnCancel.setVisibility(isPending ? View.VISIBLE : View.GONE);
            holder.btnDetail.setVisibility(View.VISIBLE);

            holder.btnPay.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onPay(item);
            });
            holder.btnCancel.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onCancel(item);
            });
            holder.btnDetail.setOnClickListener(v -> {
                if (actionListener != null) actionListener.onDetail(item);
            });
        }

        @Override
        public int getItemCount() { return items.size(); }

        // -- 状态映射 --

        private static String mapStatusLabel(String status) {
            if (status == null) return "--";
            switch (status) {
                case "待支付": return "待支付";
                case "已支付": return "已支付";
                case "已取消": return "已取消";
                case "已退款": return "已退款";
                default: return status;
            }
        }

        private static int mapStatusColor(String status) {
            if (status == null) return 0xFF9E9E9E;
            switch (status) {
                case "待支付": return 0xFFFF9800;
                case "已支付": return 0xFF4CAF50;
                case "已取消": return 0xFF9E9E9E;
                case "已退款": return 0xFF2196F3;
                default: return 0xFF9E9E9E;
            }
        }

        private static GradientDrawable createStatusBg(int color) {
            GradientDrawable bg = new GradientDrawable();
            bg.setShape(GradientDrawable.RECTANGLE);
            bg.setCornerRadius(4 * 3); // 4dp * density ≈ px
            bg.setColor(color);
            return bg;
        }

        private static String formatAmount(Double amount) {
            if (amount == null) return "";
            return "¥" + String.format("%.2f", amount);
        }

        static class VH extends RecyclerView.ViewHolder {
            TextView tvOrderNo, tvStatus, tvPackageName, tvAmount, tvTime;
            TextView btnCancel, btnPay, btnDetail;

            VH(View v) {
                super(v);
                tvOrderNo = v.findViewById(R.id.tv_order_no);
                tvStatus = v.findViewById(R.id.tv_order_status);
                tvPackageName = v.findViewById(R.id.tv_package_name);
                tvAmount = v.findViewById(R.id.tv_order_amount);
                tvTime = v.findViewById(R.id.tv_order_time);
                btnCancel = v.findViewById(R.id.btn_order_cancel);
                btnPay = v.findViewById(R.id.btn_order_pay);
                btnDetail = v.findViewById(R.id.btn_order_detail);
            }
        }
    }

    // endregion
}
