package com.pei.dehaze.ui.profile;

import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.NavController;
import androidx.navigation.NavOptions;
import androidx.navigation.Navigation;

import com.pei.dehaze.R;
import com.pei.dehaze.databinding.FragmentProfileBinding;
import com.pei.dehaze.sdk.model.user.UserInfo;
import com.pei.dehaze.ui.input.InputHistoryActivity;
import com.pei.dehaze.ui.profile.viewmodel.ProfileViewModel;
import com.pei.dehaze.ui.task.TaskListActivity;
import com.pei.dehaze.utils.ToastUtils;

import java.util.List;

public class ProfileFragment extends Fragment {

    private FragmentProfileBinding binding;
    private ProfileViewModel profileViewModel;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentProfileBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        profileViewModel = new ViewModelProvider(this).get(ProfileViewModel.class);

        buildEntryGroups();
        setupListeners();
        setupObservers();

        profileViewModel.loadUserInfo();
        profileViewModel.loadStats();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (profileViewModel != null) {
            profileViewModel.loadUserInfo();
            profileViewModel.loadStats();
        }
    }

    // ==================== 入口分组构建 ====================

    private void buildEntryGroups() {
        // 个人数据
        addEntryGroup("个人数据", new int[][]{
                {R.drawable.ic_file, R.string.entry_files, 1},
                {R.drawable.ic_dataset, R.string.entry_dataset, 2},
                {R.drawable.ic_history, R.string.entry_history, 3},
                {R.drawable.ic_favorite, R.string.entry_favorites, 4},
        });

        // 商业服务
        addEntryGroup("商业服务", new int[][]{
                {R.drawable.ic_member, R.string.entry_member, 5},
                {R.drawable.ic_package, R.string.entry_package, 6},
                {R.drawable.ic_order, R.string.entry_orders, 7},
                {R.drawable.ic_quota, R.string.entry_quota, 8},
        });

        // 其他（反馈评价 / 帮助中心 / 关于我们）
        addEntryGroup("其他", new int[][]{
                {R.drawable.ic_feedback, R.string.entry_feedback, 9},
                {R.drawable.ic_help, R.string.entry_help, 11},
                {R.drawable.ic_about, R.string.entry_about, 12},
        });
    }

    private void addEntryGroup(String title, int[][] entries) {
        View groupView = LayoutInflater.from(requireContext())
                .inflate(R.layout.item_profile_entry_group, binding.layoutEntryGroups, false);
        TextView tvTitle = groupView.findViewById(R.id.tv_group_title);
        tvTitle.setText(title);
        LinearLayout entryContainer = groupView.findViewById(R.id.layout_entries);

        for (int[] entry : entries) {
            View itemView = LayoutInflater.from(requireContext())
                    .inflate(R.layout.item_profile_entry, entryContainer, false);
            ((TextView) itemView.findViewById(R.id.tv_entry_icon)).setText(getString(entry[1]).substring(0, 1));
            ((TextView) itemView.findViewById(R.id.tv_entry_title)).setText(entry[1]);
            int action = entry[2];
            itemView.setOnClickListener(v -> onEntryClick(action));
            entryContainer.addView(itemView);
        }

        binding.layoutEntryGroups.addView(groupView);
    }

    private void onEntryClick(int action) {
        switch (action) {
            case 1: // 我的文件
                startActivity(new Intent(getActivity(), com.pei.dehaze.ui.personal.FilesActivity.class));
                break;
            case 2: // 我的数据集 → dataset
                navigateTo(R.id.datasetFragment);
                break;
            case 3: // 处理历史
                startActivity(new Intent(getActivity(), TaskListActivity.class));
                break;
            case 4: // 我的收藏
                startActivity(new Intent(getActivity(), com.pei.dehaze.ui.personal.FavoritesActivity.class));
                break;
            case 5: // 我的会员
                startActivity(new Intent(getActivity(), com.pei.dehaze.ui.personal.MemberActivity.class));
                break;
            case 6: // 我的套餐
                startActivity(new Intent(getActivity(), com.pei.dehaze.ui.personal.PackageActivity.class));
                break;
            case 7: // 我的订单
                startActivity(new Intent(getActivity(), com.pei.dehaze.ui.personal.OrdersActivity.class));
                break;
            case 8: // 我的额度
                startActivity(new Intent(getActivity(), com.pei.dehaze.ui.personal.QuotaActivity.class));
                break;
            case 9: // 反馈评价
                startActivity(new Intent(getActivity(), com.pei.dehaze.ui.personal.FeedbackActivity.class));
                break;
            case 11: // 帮助中心
                startActivity(new Intent(getActivity(), com.pei.dehaze.ui.personal.HelpActivity.class));
                break;
            case 12: // 关于我们
                startActivity(new Intent(getActivity(), com.pei.dehaze.ui.personal.AboutActivity.class));
                break;
        }
    }

    // ==================== 监听器 ====================

    private void setupListeners() {
        binding.logoutButton.setOnClickListener(v -> showLogoutConfirmDialog());
        binding.cardNotLoggedIn.setOnClickListener(v -> navigateToLogin());
        binding.cardVipBanner.setOnClickListener(v ->
                startActivity(new Intent(getActivity(), com.pei.dehaze.ui.personal.MemberActivity.class)));
    }

    // ==================== 观察者 ====================

    private void setupObservers() {
        profileViewModel.getUserInfo().observe(getViewLifecycleOwner(), userInfo -> {
            if (userInfo != null) {
                updateUserInfo(userInfo);
                buildAdminGroups(userInfo.getPerms());
            }
        });

        profileViewModel.getLoading().observe(getViewLifecycleOwner(), isLoading -> {
            boolean loading = isLoading != null && isLoading;
            binding.progressBar.setVisibility(loading ? View.VISIBLE : View.GONE);
        });

        profileViewModel.getNotLoggedIn().observe(getViewLifecycleOwner(), notLoggedIn -> {
            if (Boolean.TRUE.equals(notLoggedIn)) {
                binding.cardNotLoggedIn.setVisibility(View.VISIBLE);
                binding.layoutLoggedIn.setVisibility(View.GONE);
            } else {
                binding.cardNotLoggedIn.setVisibility(View.GONE);
                binding.layoutLoggedIn.setVisibility(View.VISIBLE);
            }
        });

        profileViewModel.getFavoriteCount().observe(getViewLifecycleOwner(), this::updateStats);
        profileViewModel.getTaskTotal().observe(getViewLifecycleOwner(), this::updateStats);
        profileViewModel.getQuotaRemaining().observe(getViewLifecycleOwner(), this::updateStats);

        profileViewModel.getError().observe(getViewLifecycleOwner(), errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                ToastUtils.showShort(requireContext(), errorMessage);
                profileViewModel.clearError();
            }
        });

        profileViewModel.getLogoutSuccess().observe(getViewLifecycleOwner(), success -> {
            if (success != null && success) {
                navigateToLogin();
            }
        });
    }

    // ==================== UI 更新 ====================

    private void updateUserInfo(UserInfo userInfo) {
        String nickname = userInfo.getNickname();
        String username = userInfo.getUsername();
        List<String> roles = userInfo.getRoles();

        binding.tvNickname.setText(nickname != null && !nickname.isEmpty()
                ? nickname : (username != null ? username : "未知用户"));
        binding.tvAvatarInitial.setText(getInitial(nickname != null ? nickname : username));

        // 角色标签
        binding.layoutRoles.removeAllViews();
        if (roles != null && !roles.isEmpty()) {
            for (String role : roles) {
                TextView tag = (TextView) LayoutInflater.from(requireContext())
                        .inflate(R.layout.item_role_tag, binding.layoutRoles, false);
                tag.setText(role.replace("ROLE_", ""));
                binding.layoutRoles.addView(tag);
            }
        }
    }

    private Long favCount = null;
    private Long taskCount = null;
    private Long quotaRemain = null;

    private void updateStats(Long value) {
        // 缓存最新值
        if (profileViewModel.getFavoriteCount().getValue() != null) {
            favCount = profileViewModel.getFavoriteCount().getValue();
        }
        if (profileViewModel.getTaskTotal().getValue() != null) {
            taskCount = profileViewModel.getTaskTotal().getValue();
        }
        if (profileViewModel.getQuotaRemaining().getValue() != null) {
            quotaRemain = profileViewModel.getQuotaRemaining().getValue();
        }
        renderStats();
    }

    private void renderStats() {
        binding.layoutStats.removeAllViews();

        addStatItem("剩余额度", quotaRemain != null ? String.valueOf(quotaRemain) : "-",
                v -> startActivity(new Intent(getActivity(), com.pei.dehaze.ui.personal.QuotaActivity.class)));
        addStatItem("本月处理", taskCount != null ? String.valueOf(taskCount) : "-",
                v -> startActivity(new Intent(getActivity(), TaskListActivity.class)));
        addStatItem("我的收藏", favCount != null ? String.valueOf(favCount) : "-",
                v -> startActivity(new Intent(getActivity(), com.pei.dehaze.ui.personal.FavoritesActivity.class)));
    }

    private void addStatItem(String label, String value, View.OnClickListener clickListener) {
        View item = LayoutInflater.from(requireContext())
                .inflate(R.layout.item_profile_stat, binding.layoutStats, false);
        ((TextView) item.findViewById(R.id.tv_stat_value)).setText(value);
        ((TextView) item.findViewById(R.id.tv_stat_label)).setText(label);
        if (clickListener != null) {
            item.setOnClickListener(clickListener);
        }
        binding.layoutStats.addView(item);
    }

    // ==================== 管理入口（权限过滤） ====================

    private void buildAdminGroups(List<String> perms) {
        // 移除之前添加的管理入口
        for (int i = binding.layoutEntryGroups.getChildCount() - 1; i >= 0; i--) {
            View child = binding.layoutEntryGroups.getChildAt(i);
            Object tag = child.getTag();
            if ("admin_group".equals(tag)) {
                binding.layoutEntryGroups.removeViewAt(i);
            }
        }

        if (perms == null || perms.isEmpty()) return;

        // 工作台（有任意管理权限即显示）
        boolean hasDashboard = hasPerm(perms, "sys:user:*");
        if (hasDashboard) {
            addAdminGroup("工作台", new int[][]{{R.string.entry_dashboard, 100}}, perms);
        }

        // 算法与数据
        addAdminGroup("算法与数据", new int[][]{
                {R.string.entry_admin_algorithm, 101},
                {R.string.entry_admin_dataset, 102},
        }, perms);

        // 系统管理
        addAdminGroup("系统管理", new int[][]{
                {R.string.entry_admin_user, 103},
                {R.string.entry_admin_role, 104},
                {R.string.entry_admin_menu, 105},
                {R.string.entry_admin_dept, 106},
                {R.string.entry_admin_dict, 107},
                {R.string.entry_admin_task, 108},
        }, perms);

        // 运营管理
        addAdminGroup("运营管理", new int[][]{
                {R.string.entry_admin_member, 109},
                {R.string.entry_admin_package, 110},
                {R.string.entry_admin_order, 111},
                {R.string.entry_admin_feedback, 112},
                {R.string.entry_admin_recommend, 113},
                {R.string.entry_admin_notify, 114},
        }, perms);
    }

    private void addAdminGroup(String title, int[][] entries, List<String> perms) {
        View groupView = LayoutInflater.from(requireContext())
                .inflate(R.layout.item_profile_entry_group, binding.layoutEntryGroups, false);
        groupView.setTag("admin_group");
        ((TextView) groupView.findViewById(R.id.tv_group_title)).setText(title);
        LinearLayout entryContainer = groupView.findViewById(R.id.layout_entries);

        boolean hasAny = false;
        for (int[] entry : entries) {
            String perm = getPermForEntry(entry[0]);
            if (perm == null || hasPerm(perms, perm)) {
                hasAny = true;
                View itemView = LayoutInflater.from(requireContext())
                        .inflate(R.layout.item_profile_entry, entryContainer, false);
                ((TextView) itemView.findViewById(R.id.tv_entry_title)).setText(entry[0]);
                int action = entry[1];
                itemView.setOnClickListener(v -> onAdminEntryClick(action));
                entryContainer.addView(itemView);
            }
        }

        if (hasAny) {
            binding.layoutEntryGroups.addView(groupView);
        }
    }

    private String getPermForEntry(int resId) {
        if (resId == R.string.entry_dashboard) return null; // 工作台用任意权限
        if (resId == R.string.entry_admin_algorithm) return "sys:algorithm:*";
        if (resId == R.string.entry_admin_dataset) return "sys:dataset:*";
        if (resId == R.string.entry_admin_user) return "sys:user:*";
        if (resId == R.string.entry_admin_role) return "sys:role:*";
        if (resId == R.string.entry_admin_menu) return "sys:menu:*";
        if (resId == R.string.entry_admin_dept) return "sys:dept:*";
        if (resId == R.string.entry_admin_dict) return "sys:dict:*";
        if (resId == R.string.entry_admin_task) return "sys:task:*";
        if (resId == R.string.entry_admin_member) return "sys:member:*";
        if (resId == R.string.entry_admin_package) return "sys:package:*";
        if (resId == R.string.entry_admin_order) return "sys:order:*";
        if (resId == R.string.entry_admin_feedback) return "sys:feedback:*";
        if (resId == R.string.entry_admin_recommend) return "sys:recommendation:*";
        if (resId == R.string.entry_admin_notify) return "sys:notify:*";
        return null;
    }

    private void onAdminEntryClick(int action) {
        Intent intent = null;
        switch (action) {
            case 100: // 工作台 → DashboardFragment
                navigateTo(R.id.dashboardFragment);
                return;
            case 101: // 算法管理
                intent = new Intent(getActivity(), com.pei.dehaze.ui.system.AlgorithmManageActivity.class);
                break;
            case 102: // 数据集管理
                intent = new Intent(getActivity(), com.pei.dehaze.ui.system.DatasetManageActivity.class);
                break;
            case 103: intent = new Intent(getActivity(), com.pei.dehaze.ui.system.UserListActivity.class); break;
            case 104: intent = new Intent(getActivity(), com.pei.dehaze.ui.system.RoleListActivity.class); break;
            case 105: intent = new Intent(getActivity(), com.pei.dehaze.ui.system.MenuListActivity.class); break;
            case 106: intent = new Intent(getActivity(), com.pei.dehaze.ui.system.DeptListActivity.class); break;
            case 107: intent = new Intent(getActivity(), com.pei.dehaze.ui.system.DictTypeListActivity.class); break;
            case 108: intent = new Intent(getActivity(), com.pei.dehaze.ui.system.TaskManageActivity.class); break;
            case 109: intent = new Intent(getActivity(), com.pei.dehaze.ui.system.MemberManageActivity.class); break;
            case 110: intent = new Intent(getActivity(), com.pei.dehaze.ui.system.PackageManageActivity.class); break;
            case 111: intent = new Intent(getActivity(), com.pei.dehaze.ui.system.OrderManageActivity.class); break;
            case 112: intent = new Intent(getActivity(), com.pei.dehaze.ui.system.FeedbackManageActivity.class); break;
            case 113: intent = new Intent(getActivity(), com.pei.dehaze.ui.system.RecommendManageActivity.class); break;
            case 114: intent = new Intent(getActivity(), com.pei.dehaze.ui.system.MessageManageActivity.class); break;
        }
        if (intent != null) startActivity(intent);
    }

    private boolean hasPerm(List<String> perms, String perm) {
        if (perms == null || perm == null) return false;
        return perms.contains(perm);
    }

    // ==================== 工具方法 ====================

    private void navigateTo(int destinationId) {
        NavController navController = Navigation.findNavController(requireActivity(),
                R.id.nav_host_fragment_content_main);
        navController.navigate(destinationId);
    }

    private String getInitial(String text) {
        if (text == null || text.isEmpty()) return "?";
        return String.valueOf(text.charAt(0)).toUpperCase();
    }

    private void showLogoutConfirmDialog() {
        new AlertDialog.Builder(requireContext())
                .setTitle("退出登录")
                .setMessage("确定要退出当前账号吗？")
                .setPositiveButton("确定", (dialog, which) -> profileViewModel.logout())
                .setNegativeButton("取消", null)
                .show();
    }

    private void navigateToLogin() {
        NavController navController = Navigation.findNavController(requireActivity(),
                R.id.nav_host_fragment_content_main);
        NavOptions options = new NavOptions.Builder()
                .setPopUpTo(R.id.nav_graph, true)
                .build();
        navController.navigate(R.id.loginFragment, null, options);
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
