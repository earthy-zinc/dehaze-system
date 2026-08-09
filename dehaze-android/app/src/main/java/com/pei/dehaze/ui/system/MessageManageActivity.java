package com.pei.dehaze.ui.system;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import androidx.annotation.NonNull;
import com.pei.dehaze.ui.common.BaseActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.pei.dehaze.databinding.ActivityManageListBinding;
import com.pei.dehaze.sdk.model.message.AnnouncementVO;
import com.pei.dehaze.ui.system.viewmodel.MessageManageViewModel;
import com.pei.dehaze.utils.ToastUtils;

import java.util.List;

/**
 * 消息管理（sys:notify:*）
 */
public class MessageManageActivity extends BaseActivity {

    private MessageManageViewModel viewModel;
    private ActivityManageListBinding binding;

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
        setupToolbar(binding.toolbar, "消息管理");

        ArrayAdapter<String> statusAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item,
                new String[]{"全部", "草稿", "已发布", "已过期"});
        statusAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        binding.spinnerStatus.setAdapter(statusAdapter);

        binding.recyclerView.setLayoutManager(new LinearLayoutManager(this));
        binding.recyclerView.setAdapter(new MessageManageAdapter());

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
        viewModel = new ViewModelProvider(this).get(MessageManageViewModel.class);
    }

    private void setupObservers() {
        viewModel.getItemList().observe(this, items -> {
            binding.recyclerView.getAdapter().notifyDataSetChanged();
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

    private class MessageManageAdapter extends RecyclerView.Adapter<MessageManageAdapter.ViewHolder> {
        @NonNull
        @Override
        public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext())
                    .inflate(android.R.layout.simple_list_item_2, parent, false);
            return new ViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
            List<?> list = viewModel.getItemList().getValue();
            if (list == null || position >= list.size()) return;
            Object obj = list.get(position);
            if (obj instanceof AnnouncementVO) {
                AnnouncementVO item = (AnnouncementVO) obj;
                holder.text1.setText(item.getTitle() != null ? item.getTitle() : "无标题");
                String type = item.getTypeLabel() != null ? item.getTypeLabel() : "";
                String status = item.getStatusLabel() != null ? item.getStatusLabel() : "";
                String sent = "已发送: " + (item.getSentCount() != null ? item.getSentCount() : 0);
                holder.text2.setText(type + "  " + status + "  " + sent);
            } else {
                holder.text1.setText(obj.toString());
                holder.text2.setText("");
            }
        }

        @Override
        public int getItemCount() {
            List<?> list = viewModel.getItemList().getValue();
            return list != null ? list.size() : 0;
        }

        class ViewHolder extends RecyclerView.ViewHolder {
            TextView text1, text2;
            ViewHolder(View itemView) {
                super(itemView);
                text1 = itemView.findViewById(android.R.id.text1);
                text2 = itemView.findViewById(android.R.id.text2);
            }
        }
    }
}
