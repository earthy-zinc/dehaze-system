package com.pei.dehaze.ui.system;

import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.pei.dehaze.R;
import com.pei.dehaze.ui.system.adapter.UserAdapter;
import com.pei.dehaze.ui.system.viewmodel.UserViewModel;
import com.pei.dehaze.sdk.model.user.UserPageVO;

import java.util.List;

public class UserListActivity extends AppCompatActivity {
    
    private UserViewModel userViewModel;
    private UserAdapter userAdapter;
    private RecyclerView recyclerView;
    private SwipeRefreshLayout swipeRefreshLayout;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_user_list);
        
        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }
    
    private void initViews() {
        recyclerView = findViewById(R.id.recycler_view);
        swipeRefreshLayout = findViewById(R.id.swipe_refresh);
        
        userAdapter = new UserAdapter();
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(userAdapter);
        
        swipeRefreshLayout.setOnRefreshListener(() -> {
            userViewModel.loadUsers();
        });
    }
    
    private void initViewModel() {
        userViewModel = new UserViewModel();
    }
    
    private void setupObservers() {
        userViewModel.getUserList().observe(this, userList -> {
            userAdapter.submitList(userList);
        });
        
        userViewModel.getLoading().observe(this, isLoading -> {
            swipeRefreshLayout.setRefreshing(isLoading != null && isLoading);
        });
    }
    
    private void loadData() {
        userViewModel.loadUsers();
    }
}