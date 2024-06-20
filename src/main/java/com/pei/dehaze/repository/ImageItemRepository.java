package com.pei.dehaze.repository;

import com.pei.dehaze.model.vo.ImageItemVO;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;

import java.util.List;

public interface ImageItemRepository extends MongoRepository<ImageItemVO, Long> {
    boolean existsByDatasetId(Long datasetId);
    Page<ImageItemVO> findByDatasetId(Long datasetId, Pageable pageable);
    Page<ImageItemVO> findByDatasetIdIn(List<Long> datasetId, Pageable pageable);


}
