package com.pei.dehaze.ui.system;

import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.pei.dehaze.R;
import com.pei.dehaze.ui.system.adapter.DeptAdapter;
import com.pei.dehaze.ui.system.viewmodel.DeptViewModel;
import com.pei.dehaze.sdk.model.dept.DeptVO;

import java.util.List;

public class DeptListActivity extends AppCompatActivity {
    
    private DeptViewModel deptViewModel;
    private DeptAdapter deptAdapter;
    private RecyclerView recyclerView;
    private SwipeRefreshLayout swipeRefreshLayout;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dept_list);
        
        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }
    
    private void initViews() {
        recyclerView = findViewById(R.id.recycler_view);
        swipeRefreshLayout = findViewById(R.id.swipe_refresh);
        
        deptAdapter = new DeptAdapter();
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(deptAdapter);
        
        swipeRefreshLayout.setOnRefreshListener(() -> {
            deptViewModel.loadDepts();
        });
    }
    
    private void initViewModel() {
        deptViewModel = new DeptViewModel();
    }
    
    private void setupObservers() {
        deptViewModel.getDeptList().observe(this, deptList -> {
            deptAdapter.submitList(deptList);
        });
        
        deptViewModel.getLoading().observe(this, isLoading -> {
            swipeRefreshLayout.setRefreshing(isLoading != null && isLoading);
        });
    }
    
    private void loadData() {
        deptViewModel.loadDepts();
    }
}