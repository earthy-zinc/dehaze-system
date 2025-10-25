package com.pei.dehaze.ui.algorithm;

import android.content.Intent;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.SearchView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.pei.dehaze.R;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.ui.algorithm.adapter.AlgorithmAdapter;
import com.pei.dehaze.ui.algorithm.viewmodel.AlgorithmViewModel;

import java.util.List;

public class AlgorithmListActivity extends AppCompatActivity {

    private AlgorithmViewModel algorithmViewModel;
    private AlgorithmAdapter algorithmAdapter;
    private RecyclerView recyclerView;
    private SwipeRefreshLayout swipeRefreshLayout;
    private SearchView searchView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_algorithm_list);

        initViews();
        initViewModel();
        setupObservers();
        loadData();
    }

    private void initViews() {
        recyclerView = findViewById(R.id.recycler_view);
        swipeRefreshLayout = findViewById(R.id.swipe_refresh);

        algorithmAdapter = new AlgorithmAdapter();
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(algorithmAdapter);

        algorithmAdapter.setOnAlgorithmClickListener(algorithm -> {
            Intent intent = new Intent(AlgorithmListActivity.this, AlgorithmDetailActivity.class);
            intent.putExtra("algorithm_id", algorithm.getId());
            startActivity(intent);
        });

        swipeRefreshLayout.setOnRefreshListener(() -> {
            loadData();
        });
    }

    private void initViewModel() {
        algorithmViewModel = new ViewModelProvider(this).get(AlgorithmViewModel.class);
    }

    private void setupObservers() {
        algorithmViewModel.getAlgorithmList().observe(this, algorithms -> {
            algorithmAdapter.submitList(algorithms);
        });

        algorithmViewModel.getLoading().observe(this, isLoading -> {
            swipeRefreshLayout.setRefreshing(isLoading != null && isLoading);
        });

        algorithmViewModel.getError().observe(this, errorMessage -> {
            if (errorMessage != null && !errorMessage.isEmpty()) {
                // 显示错误信息
            }
        });
    }

    private void loadData() {
        AlgorithmQuery query = new AlgorithmQuery();
        query.setKeywords(""); // 默认加载所有算法
        algorithmViewModel.loadAlgorithms(query);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_algorithm_list, menu);

        MenuItem searchItem = menu.findItem(R.id.action_search);
        searchView = (SearchView) searchItem.getActionView();

        searchView.setOnQueryTextListener(new SearchView.OnQueryTextListener() {
            @Override
            public boolean onQueryTextSubmit(String query) {
                searchAlgorithms(query);
                return true;
            }

            @Override
            public boolean onQueryTextChange(String newText) {
                searchAlgorithms(newText);
                return true;
            }
        });

        return true;
    }

    private void searchAlgorithms(String keywords) {
        AlgorithmQuery query = new AlgorithmQuery();
        query.setKeywords(keywords);
        algorithmViewModel.loadAlgorithms(query);
    }
}