package com.pei.dehaze.ui.system;

import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.pei.dehaze.R;
import com.pei.dehaze.ui.system.adapter.RoleAdapter;
import com.pei.dehaze.ui.system.viewmodel.RoleViewModel;
import com.pei.dehaze.sdk.model.role.RolePageVO;

import java.util.List;

public class RoleListActivity extends AppCompatActivity {
    
    private RoleViewModel roleViewModel;
    private RoleAdapter roleAdapter;
    private RecyclerView recyclerView;
    private SwipeRefreshLayout swipeRefreshLayout;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_role_list);
        
        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }
    
    private void initViews() {
        recyclerView = findViewById(R.id.recycler_view);
        swipeRefreshLayout = findViewById(R.id.swipe_refresh);
        
        roleAdapter = new RoleAdapter();
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(roleAdapter);
        
        swipeRefreshLayout.setOnRefreshListener(() -> {
            roleViewModel.loadRoles();
        });
    }
    
    private void initViewModel() {
        roleViewModel = new RoleViewModel();
    }
    
    private void setupObservers() {
        roleViewModel.getRoleList().observe(this, roleList -> {
            roleAdapter.submitList(roleList);
        });
        
        roleViewModel.getLoading().observe(this, isLoading -> {
            swipeRefreshLayout.setRefreshing(isLoading != null && isLoading);
        });
    }
    
    private void loadData() {
        roleViewModel.loadRoles();
    }
}