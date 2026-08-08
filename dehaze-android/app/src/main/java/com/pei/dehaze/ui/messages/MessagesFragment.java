package com.pei.dehaze.ui.messages;

import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.google.android.material.tabs.TabLayout;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentMessagesBinding;
import com.pei.dehaze.sdk.model.message.MessageVO;
import com.pei.dehaze.ui.messages.detail.MessagesDetailActivity;
import com.pei.dehaze.utils.ToastUtils;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MessagesFragment extends Fragment {

    private MessagesViewModel messagesViewModel;
    private UnreadMessageViewModel unreadMessageViewModel;
    private FragmentMessagesBinding binding;
    private MessagesAdapter adapter;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentMessagesBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // 消息列表 ViewModel：Fragment scope（仅本页需要）
        messagesViewModel = new ViewModelProvider(this).get(MessagesViewModel.class);
        // 未读数 ViewModel：Activity scope（全局共享，MainActivity 用于更新角标）
        unreadMessageViewModel = new ViewModelProvider(requireActivity()).get(UnreadMessageViewModel.class);

        initViews();
        setupObservers();
        messagesViewModel.loadMessages();
    }

    private void initViews() {
        // 分类 Tab
        for (String label : MessagesViewModel.getFilterLabels()) {
            binding.tabLayout.addTab(binding.tabLayout.newTab().setText(label));
        }
        binding.tabLayout.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                messagesViewModel.setFilter(tab.getPosition());
            }

            @Override
            public void onTabUnselected(TabLayout.Tab tab) {}

            @Override
            public void onTabReselected(TabLayout.Tab tab) {}
        });

        // 消息列表
        adapter = new MessagesAdapter(item -> {
            // 调后端标记已读，不阻塞跳转
            messagesViewModel.markAsRead(item.getId());
            Intent intent = new Intent(getActivity(), MessagesDetailActivity.class);
            intent.putExtra("message_title", item.getTitle());
            intent.putExtra("message_type", item.getTypeLabel() != null ? item.getTypeLabel() : item.getType());
            intent.putExtra("message_content", item.getContent() != null ? item.getContent() : item.getSummary());
            intent.putExtra("message_time", item.getCreateTime() != null ? item.getCreateTime()
                    : new SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()).format(new Date()));
            startActivity(intent);
        });
        binding.rvMessages.setLayoutManager(new LinearLayoutManager(requireContext()));
        binding.rvMessages.setAdapter(adapter);

        binding.swipeRefresh.setOnRefreshListener(() -> messagesViewModel.loadMessages());

        // 设置入口
        binding.btnSettings.setOnClickListener(v ->
                ToastUtils.showShort(getContext(), "消息设置即将上线"));

        // 全部已读
        binding.btnMarkAllRead.setOnClickListener(v -> {
            messagesViewModel.markAllRead();
            ToastUtils.showShort(getContext(), "已全部标记为已读");
        });
    }

    private void setupObservers() {
        messagesViewModel.getMessages().observe(getViewLifecycleOwner(), messages -> {
            adapter.submitList(messages);
            binding.rvMessages.scrollToPosition(0);
        });

        messagesViewModel.getRefreshing().observe(getViewLifecycleOwner(), isRefreshing -> {
            binding.swipeRefresh.setRefreshing(isRefreshing != null && isRefreshing);
        });

        messagesViewModel.getError().observe(getViewLifecycleOwner(), errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(requireContext(), errorMessage);
                messagesViewModel.clearError();
            }
        });

        // 标记单条已读成功：乐观刷新本地列表 + 触发全局未读数刷新
        messagesViewModel.getMarkedReadId().observe(getViewLifecycleOwner(), messageId -> {
            if (messageId == null) return;
            updateLocalReadStatus(messageId);
            unreadMessageViewModel.refresh();
        });

        // 全部已读成功：乐观刷新本地列表 + 触发全局未读数刷新
        messagesViewModel.getMarkedAllRead().observe(getViewLifecycleOwner(), marked -> {
            if (marked == null || !marked) return;
            markAllLocalRead();
            unreadMessageViewModel.refresh();
        });
    }

    private void updateLocalReadStatus(long messageId) {
        java.util.List<MessageVO> current = messagesViewModel.getMessages().getValue();
        if (current == null) return;
        for (MessageVO item : current) {
            if (item.getId() != null && item.getId() == messageId) {
                item.setReadStatus(1);
                break;
            }
        }
        adapter.submitList(current);
    }

    private void markAllLocalRead() {
        java.util.List<MessageVO> current = messagesViewModel.getMessages().getValue();
        if (current == null) return;
        for (MessageVO item : current) {
            item.setReadStatus(1);
        }
        adapter.submitList(current);
    }

    @Override
    public void onResume() {
        super.onResume();
        // 从详情页返回时刷新列表与未读数（详情页可能标记了已读）
        messagesViewModel.loadMessages();
        unreadMessageViewModel.refresh();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
