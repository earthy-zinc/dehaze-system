package com.pei.dehaze.ui.personal;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivitySimpleListBinding;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.OrderAPI;
import com.pei.dehaze.sdk.api.PackageAPI;
import com.pei.dehaze.sdk.model.order.OrderCreateForm;
import com.pei.dehaze.sdk.model.pkg.PackageDetailVO;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * 套餐购买 — 套餐卡片列表 + 下单
 */
public class PackageActivity extends AppCompatActivity {

    private ActivitySimpleListBinding binding;
    private PackageListViewModel viewModel;
    private PackageAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivitySimpleListBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle("商业服务");
        }

        viewModel = new ViewModelProvider(this).get(PackageListViewModel.class);
        adapter = new PackageAdapter();
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(adapter);

        binding.swipeRefresh.setOnRefreshListener(this::loadData);

        viewModel.getPackages().observe(this, list -> {
            adapter.submitList(list);
            if (list == null || list.isEmpty()) {
                binding.emptyView.setVisibility(View.VISIBLE);
                binding.emptyText.setText("暂无可用套餐");
            } else {
                binding.emptyView.setVisibility(View.GONE);
            }
        });
        viewModel.getLoading().observe(this, loading -> {
            binding.progressBar.setVisibility(loading != null && loading ? View.VISIBLE : View.GONE);
            binding.swipeRefresh.setRefreshing(loading != null && loading);
        });
        viewModel.getError().observe(this, msg -> {
            if (msg != null && !msg.isEmpty()) ToastUtils.showShort(this, msg);
        });

        loadData();
    }

    private void loadData() {
        PackageAPI.listOnSale(RepositoryAdapters.wrap(viewModel.createLoadingCallback(packages ->
                viewModel.setPackages(packages != null ? packages : new ArrayList<>()))));
    }

    private void buyPackage(PackageDetailVO pkg) {
        OrderCreateForm form = new OrderCreateForm();
        form.setPackageId(pkg.getId());
        OrderAPI.create(form, RepositoryAdapters.wrap(new RepositoryCallback<com.pei.dehaze.sdk.model.order.PayResult>() {
            @Override
            public void onSuccess(com.pei.dehaze.sdk.model.order.PayResult data) {
                ToastUtils.showShort(PackageActivity.this, "订单已创建，请前往我的订单支付");
            }

            @Override
            public void onError(String errorMessage) {
                ToastUtils.showShort(PackageActivity.this, errorMessage);
            }
        }));
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            finish();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    public static class PackageListViewModel extends BaseViewModel {
        private final androidx.lifecycle.MutableLiveData<List<PackageDetailVO>> packages =
                new androidx.lifecycle.MutableLiveData<>();

        public androidx.lifecycle.LiveData<List<PackageDetailVO>> getPackages() {
            return packages;
        }

        public void setPackages(List<PackageDetailVO> list) {
            packages.postValue(list);
        }

        public <T> RepositoryCallback<T> createLoadingCallback(OnSuccess<T> onSuccess) {
            return withLoading(onSuccess);
        }
    }

    class PackageAdapter extends RecyclerView.Adapter<PackageAdapter.VH> {
        private List<PackageDetailVO> items = new ArrayList<>();

        void submitList(List<PackageDetailVO> newItems) {
            items = newItems != null ? newItems : new ArrayList<>();
            notifyDataSetChanged();
        }

        @NonNull
        @Override
        public VH onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View v = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_package, parent, false);
            return new VH(v);
        }

        @Override
        public void onBindViewHolder(@NonNull VH holder, int position) {
            PackageDetailVO item = items.get(position);
            holder.name.setText(item.getName() != null ? item.getName() : "未命名套餐");

            if (item.getSalePrice() != null) {
                holder.price.setText("¥" + String.format("%.2f", item.getSalePrice()));
            } else {
                holder.price.setText("价格待定");
            }

            // 原价（划线展示）
            if (item.getOriginalPrice() != null && item.getOriginalPrice() > 0
                    && !item.getOriginalPrice().equals(item.getSalePrice())) {
                holder.originalPrice.setText("¥" + String.format("%.2f", item.getOriginalPrice()));
                holder.originalPrice.setPaintFlags(
                        holder.originalPrice.getPaintFlags() | android.graphics.Paint.STRIKE_THRU_TEXT_FLAG);
                holder.originalPrice.setVisibility(View.VISIBLE);
            } else {
                holder.originalPrice.setVisibility(View.GONE);
            }

            holder.period.setText(item.getPeriod() != null ? item.getPeriod() : "");

            String desc = item.getDescription();
            if (desc != null && !desc.isEmpty()) {
                holder.description.setText(desc);
                holder.description.setVisibility(View.VISIBLE);
            } else {
                holder.description.setVisibility(View.GONE);
            }

            String level = item.getLevelName();
            if (level != null && !level.isEmpty()) {
                holder.level.setText("等级: " + level);
                holder.level.setVisibility(View.VISIBLE);
            } else {
                holder.level.setVisibility(View.GONE);
            }

            holder.btnBuy.setOnClickListener(v -> buyPackage(item));
        }

        @Override
        public int getItemCount() {
            return items.size();
        }

        class VH extends RecyclerView.ViewHolder {
            TextView name, price, originalPrice, period, description, level;
            Button btnBuy;

            VH(View v) {
                super(v);
                name = v.findViewById(R.id.tv_package_name);
                price = v.findViewById(R.id.tv_price);
                originalPrice = v.findViewById(R.id.tv_original_price);
                period = v.findViewById(R.id.tv_period);
                description = v.findViewById(R.id.tv_description);
                level = v.findViewById(R.id.tv_level);
                btnBuy = v.findViewById(R.id.btn_buy);
            }
        }
    }
}
