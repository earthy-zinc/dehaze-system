package com.pei.dehaze;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;
import androidx.navigation.fragment.NavHostFragment;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.pei.dehaze.databinding.ActivityMainBinding;
import com.pei.dehaze.ui.compare.CompareActivity;
import com.pei.dehaze.ui.evaluation.EvaluationActivity;
import com.pei.dehaze.ui.presentation.PresentationActivity;

public class MainActivity extends AppCompatActivity {

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);

        // 通过FragmentContainerView获取NavController
        NavHostFragment navHostFragment = (NavHostFragment) getSupportFragmentManager()
                .findFragmentById(R.id.nav_host_fragment_content_main);
        NavController navController = navHostFragment.getNavController();

        // 设置底部导航栏（仅包含Fragment导航）
        // 注意：我们现在使用的是新的视图ID
        BottomNavigationView bottomNavigationView = findViewById(R.id.bottom_navigation);
        NavigationUI.setupWithNavController(bottomNavigationView, navController);
        
        // 处理启动Activity的按钮
        Button btnCompare = findViewById(R.id.btn_compare);
        Button btnEvaluation = findViewById(R.id.btn_evaluation);
        Button btnPresentation = findViewById(R.id.btn_presentation);
        
        btnCompare.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(MainActivity.this, CompareActivity.class));
            }
        });
        
        btnEvaluation.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(MainActivity.this, EvaluationActivity.class));
            }
        });
        
        btnPresentation.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(MainActivity.this, PresentationActivity.class));
            }
        });

        appBarConfiguration = new AppBarConfiguration.Builder(
                R.id.dashboardFragment, 
                R.id.datasetFragment, 
                R.id.algorithmFragment, 
                R.id.systemManagementFragment)
                .build();
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);
    }

    @Override
    public boolean onSupportNavigateUp() {
        // 通过FragmentContainerView获取NavController
        NavHostFragment navHostFragment = (NavHostFragment) getSupportFragmentManager()
                .findFragmentById(R.id.nav_host_fragment_content_main);
        NavController navController = navHostFragment.getNavController();

        return NavigationUI.navigateUp(navController, appBarConfiguration)
                || super.onSupportNavigateUp();
    }
}