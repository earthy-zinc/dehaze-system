package com.pei.dehaze.ui.compare;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import androidx.viewpager2.widget.ViewPager2;

import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;
import com.pei.dehaze.R;

public class CompareActivity extends AppCompatActivity {

    private ViewPager2 viewPager;
    private TabLayout tabLayout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_compare);

        initViews();
    }

    private void initViews() {
        viewPager = findViewById(R.id.view_pager);
        tabLayout = findViewById(R.id.tab_layout);

        ComparePagerAdapter adapter = new ComparePagerAdapter(this);
        viewPager.setAdapter(adapter);

        new TabLayoutMediator(tabLayout, viewPager,
                (tab, position) -> {
                    switch (position) {
                        case 0:
                            tab.setText("并排对比");
                            break;
                        case 1:
                            tab.setText("重叠对比");
                            break;
                        default:
                            tab.setText("对比");
                            break;
                    }
                }).attach();
    }

    private static class ComparePagerAdapter extends FragmentStateAdapter {

        public ComparePagerAdapter(FragmentActivity fa) {
            super(fa);
        }

        @Override
        public Fragment createFragment(int position) {
            switch (position) {
                case 0:
                    return new ParallelFragment();
                case 1:
                    return new OverlapFragment();
                default:
                    return new ParallelFragment();
            }
        }

        @Override
        public int getItemCount() {
            return 2; // 两个对比模式
        }
    }
}