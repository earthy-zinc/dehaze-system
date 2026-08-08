package com.pei.dehaze;

import android.os.Bundle;
import android.view.View;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.NavController;
import androidx.navigation.NavOptions;
import androidx.navigation.fragment.NavHostFragment;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.google.android.material.badge.BadgeDrawable;
import com.pei.dehaze.databinding.ActivityMainBinding;
import com.pei.dehaze.ui.messages.UnreadMessageViewModel;

public class MainActivity extends AppCompatActivity implements DehazeApplication.SessionInvalidHandler {

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;
    private UnreadMessageViewModel unreadMessageViewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);

        NavHostFragment navHostFragment = (NavHostFragment) getSupportFragmentManager()
                .findFragmentById(R.id.nav_host_fragment_content_main);
        NavController navController = navHostFragment.getNavController();

        // 5 个新顶级目的地：首页/工具/去雾/消息/我的
        appBarConfiguration = new AppBarConfiguration.Builder(
                R.id.homeFragment,
                R.id.toolsFragment,
                R.id.dehazeFragment,
                R.id.messagesFragment,
                R.id.profileFragment)
                .build();
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);

        // L0/L2 显隐控制：认证页隐藏 Toolbar 和 TabBar，二级页隐藏 TabBar
        navController.addOnDestinationChangedListener((controller, destination, arguments) -> {
            boolean isTopLevel = appBarConfiguration.getTopLevelDestinations().contains(destination.getId());
            boolean isAuth = destination.getId() == R.id.loginFragment
                    || destination.getId() == R.id.registerFragment;

            // L0 认证页隐藏 Toolbar 和 BottomNavigation
            if (isAuth) {
                binding.toolbar.setVisibility(View.GONE);
                if (getSupportActionBar() != null) {
                    getSupportActionBar().hide();
                }
            } else {
                binding.toolbar.setVisibility(View.VISIBLE);
                if (getSupportActionBar() != null) {
                    getSupportActionBar().show();
                }
            }

            // 非顶级目的地（L2/L3）和认证页隐藏 TabBar
            binding.bottomNavigation.setVisibility((isAuth || !isTopLevel) ? View.GONE : View.VISIBLE);
        });

        // 关联 BottomNavigationView 与 NavController
        NavigationUI.setupWithNavController(binding.bottomNavigation, navController);

        // 全局未读消息数：Activity scope 持有，observe 后更新消息 Tab 角标
        unreadMessageViewModel = new ViewModelProvider(this).get(UnreadMessageViewModel.class);
        unreadMessageViewModel.getUnreadCount().observe(this, count -> updateMessagesBadge(count != null ? count : 0));
    }

    @Override
    protected void onResume() {
        super.onResume();
        // 每次回到 MainActivity（从详情页返回、从其他 Tab 切回）刷新未读数
        unreadMessageViewModel.refresh();
    }

    private void updateMessagesBadge(int count) {
        BadgeDrawable badge = binding.bottomNavigation.getOrCreateBadge(R.id.messagesFragment);
        badge.setNumber(count);
        badge.setVisible(count > 0);
    }

    @Override
    public boolean onSupportNavigateUp() {
        NavHostFragment navHostFragment = (NavHostFragment) getSupportFragmentManager()
                .findFragmentById(R.id.nav_host_fragment_content_main);
        NavController navController = navHostFragment.getNavController();

        return NavigationUI.navigateUp(navController, appBarConfiguration)
                || super.onSupportNavigateUp();
    }

    @Override
    public void onSessionInvalid() {
        if (isFinishing() || isDestroyed()) {
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("登录已失效")
                .setMessage("您的登录状态已过期，请重新登录")
                .setCancelable(false)
                .setPositiveButton("重新登录", (dialog, which) -> navigateToLogin())
                .show();
    }

    private void navigateToLogin() {
        NavHostFragment navHostFragment = (NavHostFragment) getSupportFragmentManager()
                .findFragmentById(R.id.nav_host_fragment_content_main);
        if (navHostFragment == null) {
            return;
        }
        NavController navController = navHostFragment.getNavController();
        NavOptions options = new NavOptions.Builder()
                .setPopUpTo(R.id.nav_graph, true)
                .build();
        navController.navigate(R.id.loginFragment, null, options);
    }
}