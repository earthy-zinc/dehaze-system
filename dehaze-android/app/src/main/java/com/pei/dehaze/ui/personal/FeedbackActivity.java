package com.pei.dehaze.ui.personal;

import android.app.Dialog;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import androidx.viewpager2.widget.ViewPager2;

import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;
import com.pei.dehaze.R;
import com.pei.dehaze.databinding.ActivitySimpleListBinding;
import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.repository.RepositoryCallback;
import com.pei.dehaze.sdk.api.FeedbackAPI;
import com.pei.dehaze.sdk.model.PageResult;
import com.pei.dehaze.sdk.model.feedback.FeedbackCreateForm;
import com.pei.dehaze.sdk.model.feedback.FeedbackPageVO;
import com.pei.dehaze.sdk.model.feedback.MyRatingVO;
import com.pei.dehaze.ui.common.BaseActivity;
import com.pei.dehaze.ui.common.BaseViewModel;
import com.pei.dehaze.ui.common.BaseActivity;
import com.pei.dehaze.utils.ToastUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * 反馈评价 — 双Tab（我的反馈 / 我的评价）+ 新增反馈
 */
public class FeedbackActivity extends BaseActivity {

    private TabLayout tabLayout;
    private ViewPager2 viewPager;
    private com.google.android.material.floatingactionbutton.FloatingActionButton fabAdd;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_feedback);

        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle("反馈评价");
        }

        tabLayout = findViewById(R.id.tab_layout);
        viewPager = findViewById(R.id.view_pager);
        fabAdd = findViewById(R.id.fab_add);

        viewPager.setAdapter(new FeedbackPagerAdapter(this));
        new TabLayoutMediator(tabLayout, viewPager, (tab, position) -> {
            tab.setText(position == 0 ? "我的反馈" : "我的评价");
        }).attach();

        fabAdd.setOnClickListener(v -> showAddFeedbackDialog());
    }

    private void showAddFeedbackDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        View dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_add_feedback, null);
        builder.setView(dialogView);
        builder.setTitle("新增反馈");

        EditText etTitle = dialogView.findViewById(R.id.et_title);
        EditText etContent = dialogView.findViewById(R.id.et_content);
        Spinner spType = dialogView.findViewById(R.id.sp_type);

        String[] types = {"建议", "问题", "投诉", "其他"};
        ArrayAdapter<String> typeAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, types);
        typeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spType.setAdapter(typeAdapter);

        Dialog dialog = builder.create();

        Button btnSubmit = dialogView.findViewById(R.id.btn_submit);
        Button btnCancel = dialogView.findViewById(R.id.btn_cancel);

        btnSubmit.setOnClickListener(v -> {
            String title = etTitle.getText().toString().trim();
            String content = etContent.getText().toString().trim();
            if (title.isEmpty()) {
                ToastUtils.showShort(this, "请输入标题");
                return;
            }
            if (content.isEmpty()) {
                ToastUtils.showShort(this, "请输入内容");
                return;
            }

            FeedbackCreateForm form = new FeedbackCreateForm();
            form.setTitle(title);
            form.setContent(content);
            form.setFeedbackType(spType.getSelectedItem().toString());

            FeedbackAPI.createFeedback(form, RepositoryAdapters.wrap(new RepositoryCallback<Object>() {
                @Override
                public void onSuccess(Object data) {
                    ToastUtils.showShort(FeedbackActivity.this, "反馈已提交");
                    dialog.dismiss();
                }

                @Override
                public void onError(String errorMessage) {
                    ToastUtils.showShort(FeedbackActivity.this, errorMessage);
                }
            }));
        });

        btnCancel.setOnClickListener(v -> dialog.dismiss());
        dialog.show();
    }

    // ---- ViewPager Adapter ----

    static class FeedbackPagerAdapter extends FragmentStateAdapter {

        FeedbackPagerAdapter(@NonNull AppCompatActivity activity) {
            super(activity);
        }

        @NonNull
        @Override
        public androidx.fragment.app.Fragment createFragment(int position) {
            if (position == 0) {
                return new FeedbackListFragment();
            } else {
                return new RatingListFragment();
            }
        }

        @Override
        public int getItemCount() {
            return 2;
        }
    }

    // ---- 我的反馈 Fragment ----

    public static class FeedbackListFragment extends androidx.fragment.app.Fragment {

        private ActivitySimpleListBinding binding;
        private FeedbackListViewModel viewModel;
        private FeedbackAdapter adapter;
        private int currentPage = 1;
        private static final int PAGE_SIZE = 20;
        private boolean isLoading = false;
        private boolean hasMore = true;

        @Override
        public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container,
                                 Bundle savedInstanceState) {
            binding = ActivitySimpleListBinding.inflate(inflater, container, false);
            return binding.getRoot();
        }

        @Override
        public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
            viewModel = new ViewModelProvider(this).get(FeedbackListViewModel.class);
            adapter = new FeedbackAdapter();
            binding.recyclerView.setLayoutManager(new LinearLayoutManager(requireContext()));
            binding.recyclerView.setAdapter(adapter);

            binding.recyclerView.addOnScrollListener(new RecyclerView.OnScrollListener() {
                @Override
                public void onScrolled(@NonNull RecyclerView recyclerView, int dx, int dy) {
                    LinearLayoutManager lm = (LinearLayoutManager) recyclerView.getLayoutManager();
                    if (lm != null && lm.findLastVisibleItemPosition() + 1 >= adapter.getItemCount()
                            && hasMore && !isLoading) {
                        loadMore();
                    }
                }
            });

            binding.swipeRefresh.setOnRefreshListener(() -> {
                currentPage = 1;
                hasMore = true;
                loadData();
            });

            viewModel.getFeedbacks().observe(getViewLifecycleOwner(), list -> {
                adapter.submitList(list);
                if (list == null || list.isEmpty()) {
                    binding.emptyView.setVisibility(View.VISIBLE);
                    binding.emptyText.setText("暂无反馈记录");
                } else {
                    binding.emptyView.setVisibility(View.GONE);
                }
            });
            viewModel.getLoading().observe(getViewLifecycleOwner(), loading -> {
                isLoading = loading != null && loading;
                binding.swipeRefresh.setRefreshing(isLoading);
                binding.progressBar.setVisibility(isLoading ? View.VISIBLE : View.GONE);
            });
            viewModel.getError().observe(getViewLifecycleOwner(), msg -> {
                if (msg != null && !msg.isEmpty()) ToastUtils.showShort(requireContext(), msg);
            });

            loadData();
        }

        private void loadData() {
            FeedbackAPI.listMyFeedback(currentPage, PAGE_SIZE,
                    RepositoryAdapters.wrap(viewModel.createLoadingCallback(data -> {
                        List<FeedbackPageVO> list = data.getList();
                        viewModel.setFeedbacks(list != null ? list : new ArrayList<>());
                        hasMore = list != null && list.size() >= PAGE_SIZE;
                    })));
        }

        private void loadMore() {
            currentPage++;
            FeedbackAPI.listMyFeedback(currentPage, PAGE_SIZE,
                    RepositoryAdapters.wrap(viewModel.createLoadingCallback(data -> {
                        List<FeedbackPageVO> list = data.getList();
                        if (list != null) {
                            adapter.addAll(list);
                        }
                        hasMore = list != null && list.size() >= PAGE_SIZE;
                    })));
        }

        public static class FeedbackListViewModel extends BaseViewModel {
            private final androidx.lifecycle.MutableLiveData<List<FeedbackPageVO>> feedbacks =
                    new androidx.lifecycle.MutableLiveData<>();

            public androidx.lifecycle.LiveData<List<FeedbackPageVO>> getFeedbacks() {
                return feedbacks;
            }

            public void setFeedbacks(List<FeedbackPageVO> list) {
                feedbacks.postValue(list);
            }

            public <T> RepositoryCallback<T> createLoadingCallback(OnSuccess<T> onSuccess) {
                return withLoading(onSuccess);
            }
        }

        static class FeedbackAdapter extends RecyclerView.Adapter<FeedbackAdapter.VH> {
            private final List<FeedbackPageVO> items = new ArrayList<>();

            void submitList(List<FeedbackPageVO> newItems) {
                items.clear();
                if (newItems != null) items.addAll(newItems);
                notifyDataSetChanged();
            }

            void addAll(List<FeedbackPageVO> newItems) {
                if (newItems != null) {
                    int start = items.size();
                    items.addAll(newItems);
                    notifyItemRangeInserted(start, newItems.size());
                }
            }

            @NonNull
            @Override
            public VH onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
                View v = LayoutInflater.from(parent.getContext())
                        .inflate(android.R.layout.simple_list_item_2, parent, false);
                return new VH(v);
            }

            @Override
            public void onBindViewHolder(@NonNull VH holder, int position) {
                FeedbackPageVO item = items.get(position);
                holder.title.setText(item.getTitle() != null ? item.getTitle() : "无标题");
                String status = item.getStatus();
                String type = item.getFeedbackType();
                String subtitle = (status != null ? "状态: " + status : "")
                        + (type != null ? "  类型: " + type : "");
                holder.subtitle.setText(subtitle.trim());
            }

            @Override
            public int getItemCount() {
                return items.size();
            }

            static class VH extends RecyclerView.ViewHolder {
                TextView title, subtitle;

                VH(View v) {
                    super(v);
                    title = v.findViewById(android.R.id.text1);
                    subtitle = v.findViewById(android.R.id.text2);
                }
            }
        }
    }

    // ---- 我的评价 Fragment ----

    public static class RatingListFragment extends androidx.fragment.app.Fragment {

        private ActivitySimpleListBinding binding;
        private RatingListViewModel viewModel;
        private RatingAdapter adapter;
        private int currentPage = 1;
        private static final int PAGE_SIZE = 20;
        private boolean isLoading = false;
        private boolean hasMore = true;

        @Override
        public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container,
                                 Bundle savedInstanceState) {
            binding = ActivitySimpleListBinding.inflate(inflater, container, false);
            return binding.getRoot();
        }

        @Override
        public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
            viewModel = new ViewModelProvider(this).get(RatingListViewModel.class);
            adapter = new RatingAdapter();
            binding.recyclerView.setLayoutManager(new LinearLayoutManager(requireContext()));
            binding.recyclerView.setAdapter(adapter);

            binding.recyclerView.addOnScrollListener(new RecyclerView.OnScrollListener() {
                @Override
                public void onScrolled(@NonNull RecyclerView recyclerView, int dx, int dy) {
                    LinearLayoutManager lm = (LinearLayoutManager) recyclerView.getLayoutManager();
                    if (lm != null && lm.findLastVisibleItemPosition() + 1 >= adapter.getItemCount()
                            && hasMore && !isLoading) {
                        loadMore();
                    }
                }
            });

            binding.swipeRefresh.setOnRefreshListener(() -> {
                currentPage = 1;
                hasMore = true;
                loadData();
            });

            viewModel.getRatings().observe(getViewLifecycleOwner(), list -> {
                adapter.submitList(list);
                if (list == null || list.isEmpty()) {
                    binding.emptyView.setVisibility(View.VISIBLE);
                    binding.emptyText.setText("暂无评价记录");
                } else {
                    binding.emptyView.setVisibility(View.GONE);
                }
            });
            viewModel.getLoading().observe(getViewLifecycleOwner(), loading -> {
                isLoading = loading != null && loading;
                binding.swipeRefresh.setRefreshing(isLoading);
                binding.progressBar.setVisibility(isLoading ? View.VISIBLE : View.GONE);
            });
            viewModel.getError().observe(getViewLifecycleOwner(), msg -> {
                if (msg != null && !msg.isEmpty()) ToastUtils.showShort(requireContext(), msg);
            });

            loadData();
        }

        private void loadData() {
            FeedbackAPI.listMyRatings(currentPage, PAGE_SIZE,
                    RepositoryAdapters.wrap(viewModel.createLoadingCallback(data -> {
                        List<MyRatingVO> list = data.getList();
                        viewModel.setRatings(list != null ? list : new ArrayList<>());
                        hasMore = list != null && list.size() >= PAGE_SIZE;
                    })));
        }

        private void loadMore() {
            currentPage++;
            FeedbackAPI.listMyRatings(currentPage, PAGE_SIZE,
                    RepositoryAdapters.wrap(viewModel.createLoadingCallback(data -> {
                        List<MyRatingVO> list = data.getList();
                        if (list != null) {
                            adapter.addAll(list);
                        }
                        hasMore = list != null && list.size() >= PAGE_SIZE;
                    })));
        }

        public static class RatingListViewModel extends BaseViewModel {
            private final androidx.lifecycle.MutableLiveData<List<MyRatingVO>> ratings =
                    new androidx.lifecycle.MutableLiveData<>();

            public androidx.lifecycle.LiveData<List<MyRatingVO>> getRatings() {
                return ratings;
            }

            public void setRatings(List<MyRatingVO> list) {
                ratings.postValue(list);
            }

            public <T> RepositoryCallback<T> createLoadingCallback(OnSuccess<T> onSuccess) {
                return withLoading(onSuccess);
            }
        }

        static class RatingAdapter extends RecyclerView.Adapter<RatingAdapter.VH> {
            private final List<MyRatingVO> items = new ArrayList<>();

            void submitList(List<MyRatingVO> newItems) {
                items.clear();
                if (newItems != null) items.addAll(newItems);
                notifyDataSetChanged();
            }

            void addAll(List<MyRatingVO> newItems) {
                if (newItems != null) {
                    int start = items.size();
                    items.addAll(newItems);
                    notifyItemRangeInserted(start, newItems.size());
                }
            }

            @NonNull
            @Override
            public VH onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
                View v = LayoutInflater.from(parent.getContext())
                        .inflate(android.R.layout.simple_list_item_2, parent, false);
                return new VH(v);
            }

            @Override
            public void onBindViewHolder(@NonNull VH holder, int position) {
                MyRatingVO item = items.get(position);
                holder.title.setText(item.getAlgorithmName() != null ? item.getAlgorithmName() : "未命名算法");
                Integer rating = item.getRating();
                String comment = item.getComment();
                String subtitle = "评分: " + (rating != null ? rating + "星" : "未评分")
                        + (comment != null && !comment.isEmpty() ? "  " + comment : "");
                holder.subtitle.setText(subtitle);
            }

            @Override
            public int getItemCount() {
                return items.size();
            }

            static class VH extends RecyclerView.ViewHolder {
                TextView title, subtitle;

                VH(View v) {
                    super(v);
                    title = v.findViewById(android.R.id.text1);
                    subtitle = v.findViewById(android.R.id.text2);
                }
            }
        }
    }
}
