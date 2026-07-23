package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.algorithm.Algorithm;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmFavorite;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmQuery;
import com.pei.dehaze.sdk.model.algorithm.AlgorithmStatus;

import java.util.List;

public class AlgorithmRepository {

    public void getAlgorithms(AlgorithmQuery query, RepositoryCallback<List<Algorithm>> callback) {
        AlgorithmAPI.getList(query, RepositoryAdapters.wrap(callback));
    }

    public void getAlgorithmDetail(long id, RepositoryCallback<Algorithm> callback) {
        AlgorithmAPI.getAlgorithmInfoById(id, RepositoryAdapters.wrap(callback));
    }

    public void compare(String ids, RepositoryCallback<List<Algorithm>> callback) {
        AlgorithmAPI.compare(ids, RepositoryAdapters.wrap(callback));
    }

    public void getOptions(RepositoryCallback<List<Option>> callback) {
        AlgorithmAPI.getOption(RepositoryAdapters.wrap(callback));
    }

    public void listFavorites(RepositoryCallback<List<AlgorithmFavorite>> callback) {
        AlgorithmAPI.listFavorites(RepositoryAdapters.wrap(callback));
    }

    public void toggleFavorite(long id, RepositoryCallback<Void> callback) {
        AlgorithmAPI.toggleFavorite(id, RepositoryAdapters.wrap(callback));
    }

    public void addAlgorithm(Algorithm data, RepositoryCallback<Void> callback) {
        AlgorithmAPI.add(data, RepositoryAdapters.wrap(callback));
    }

    public void updateAlgorithm(long id, Algorithm data, RepositoryCallback<Void> callback) {
        AlgorithmAPI.update(id, data, RepositoryAdapters.wrap(callback));
    }

    public void updateAlgorithmStatus(long id, AlgorithmStatus status, RepositoryCallback<Void> callback) {
        AlgorithmAPI.updateStatus(id, status, RepositoryAdapters.wrap(callback));
    }

    public void deleteAlgorithms(List<Long> ids, RepositoryCallback<Void> callback) {
        AlgorithmAPI.deleteByIds(ids, RepositoryAdapters.wrap(callback));
    }
}
