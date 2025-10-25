package com.pei.dehaze.ui.dataset;

import android.util.Log;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.pei.dehaze.sdk.ApiCallback;
import com.pei.dehaze.sdk.api.DatasetAPI;
import com.pei.dehaze.sdk.network.ApiException;
import com.pei.dehaze.sdk.model.dataset.Dataset;
import com.pei.dehaze.sdk.model.dataset.ImageItem;
import com.pei.dehaze.sdk.model.dataset.ImageItemQuery;
import com.pei.dehaze.sdk.model.dataset.ImageUrl;
import com.pei.dehaze.ui.dataset.model.ImageType;
import com.pei.dehaze.ui.dataset.model.ViewCard;

import java.util.ArrayList;
import java.util.List;

public class DatasetViewModel extends ViewModel {
    private static final String TAG = "DatasetViewModel";

    // 数据集信息
    private MutableLiveData<Dataset> datasetInfo = new MutableLiveData<>();
    
    // 图片列表
    private MutableLiveData<List<ViewCard>> images = new MutableLiveData<>();
    
    // 图片类型列表
    private MutableLiveData<List<ImageType>> imageTypes = new MutableLiveData<>();
    
    // 当前选中的图片类型
    private MutableLiveData<ImageType> curImageType = new MutableLiveData<>();
    
    // 加载状态
    private MutableLiveData<Boolean> loading = new MutableLiveData<>();
    
    // 错误信息
    private MutableLiveData<String> error = new MutableLiveData<>();
    
    // 总页数
    private MutableLiveData<Integer> totalPages = new MutableLiveData<>();
    
    // 当前页码
    private int currentPage = 1;
    
    // 每页大小
    private int pageSize = 10;
    
    // 数据集ID
    private int datasetId;
    
    // 图片数据缓存
    private List<ImageItem> imageData = new ArrayList<>();

    public DatasetViewModel() {
        images.setValue(new ArrayList<>());
        imageTypes.setValue(new ArrayList<>());
        loading.setValue(false);
        error.setValue("");
        totalPages.setValue(1);
        datasetInfo.setValue(new Dataset());
    }

    public LiveData<Dataset> getDatasetInfo() {
        return datasetInfo;
    }

    public LiveData<List<ViewCard>> getImages() {
        return images;
    }

    public LiveData<List<ImageType>> getImageTypes() {
        return imageTypes;
    }

    public LiveData<ImageType> getCurImageType() {
        return curImageType;
    }

    public LiveData<Boolean> getLoading() {
        return loading;
    }

    public LiveData<String> getError() {
        return error;
    }

    public LiveData<Integer> getTotalPages() {
        return totalPages;
    }

    /**
     * 设置数据集ID
     * @param datasetId 数据集ID
     */
    public void setDatasetId(int datasetId) {
        this.datasetId = datasetId;
    }

    /**
     * 加载数据集信息
     */
    public void loadDatasetInfo() {
        if (datasetId <= 0) {
            error.setValue("数据集ID无效");
            return;
        }

        loading.setValue(true);
        DatasetAPI.getDatasetInfoById(datasetId, new ApiCallback<Dataset>() {
            @Override
            public void onSuccess(Dataset data) {
                loading.postValue(false);
                datasetInfo.postValue(data);
            }

            @Override
            public void onError(int code, String message) {
                loading.postValue(false);
                error.postValue("获取数据集信息失败: " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                loading.postValue(false);
                error.postValue("网络错误: " + e.getMessage());
            }
        });
    }

    /**
     * 加载图片数据
     */
    public void loadImages() {
        if (datasetId <= 0) {
            error.setValue("数据集ID无效");
            return;
        }

        loading.setValue(true);
        
        ImageItemQuery query = new ImageItemQuery();
        query.setPageNum(currentPage);
        query.setPageSize(pageSize);
        
        DatasetAPI.getImageItem(datasetId, query, new ApiCallback<List<ImageItem>>() {
            @Override
            public void onSuccess(List<ImageItem> data) {
                loading.postValue(false);
                
                // 更新图片数据
                if (currentPage == 1) {
                    imageData.clear();
                }
                imageData.addAll(data);
                
                // 更新总页数
                int total = datasetInfo.getValue() != null ? datasetInfo.getValue().getTotal() : 0;
                totalPages.postValue((int) Math.ceil((double) total / pageSize));
                
                // 处理图片类型
                processImageTypes();
                
                // 切换图片URL
                switchImageUrl();
            }

            @Override
            public void onError(int code, String message) {
                loading.postValue(false);
                error.postValue("获取图片数据失败: " + message);
            }

            @Override
            public void onFailure(ApiException e) {
                loading.postValue(false);
                error.postValue("网络错误: " + e.getMessage());
            }
        });
    }

    /**
     * 处理图片类型
     */
    private void processImageTypes() {
        if (imageData.isEmpty()) {
            return;
        }

        List<ImageType> types = new ArrayList<>();
        List<ImageUrl> urls = imageData.get(0).getImgUrl();
        
        for (int i = 0; i < urls.size(); i++) {
            ImageUrl url = urls.get(i);
            types.add(new ImageType(i, url.getType(), i == 0));
        }
        
        imageTypes.postValue(types);
        
        // 设置当前选中的图片类型
        for (ImageType type : types) {
            if (type.isEnabled()) {
                curImageType.postValue(type);
                break;
            }
        }
    }

    /**
     * 切换图片URL
     */
    private void switchImageUrl() {
        ImageType currentType = curImageType.getValue();
        if (currentType == null || imageData.isEmpty()) {
            images.postValue(new ArrayList<>());
            return;
        }

        List<ViewCard> viewCards = new ArrayList<>();
        for (ImageItem item : imageData) {
            List<ImageUrl> urls = item.getImgUrl();
            if (currentType.getId() < urls.size()) {
                ImageUrl url = urls.get(currentType.getId());
                viewCards.add(new ViewCard(
                    item.getId(),
                    url.getUrl(),
                    url.getOriginUrl() != null ? url.getOriginUrl() : url.getUrl(),
                    url.getDescription() != null ? url.getDescription() : ""
                ));
            }
        }
        
        images.postValue(viewCards);
    }

    /**
     * 切换图片类型
     * @param typeId 图片类型ID
     */
    public void switchImageType(int typeId) {
        List<ImageType> types = imageTypes.getValue();
        if (types == null) {
            return;
        }

        for (ImageType type : types) {
            type.setEnabled(type.getId() == typeId);
            if (type.isEnabled()) {
                curImageType.postValue(type);
            }
        }
        
        imageTypes.postValue(types);
        switchImageUrl();
    }

    /**
     * 加载更多数据
     */
    public void loadMore() {
        if (currentPage < totalPages.getValue()) {
            currentPage++;
            loadImages();
        }
    }

    /**
     * 重置查询
     */
    public void resetQuery() {
        currentPage = 1;
        loadImages();
    }

    /**
     * 搜索图片
     * @param keywords 关键词
     */
    public void searchImages(String keywords) {
        // TODO: 实现搜索功能
        // 目前只是重新加载第一页数据
        currentPage = 1;
        loadImages();
    }
}