package com.pei.dehaze.repository;

import com.pei.dehaze.model.entity.LoginLog;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface LoginLogRepository extends MongoRepository<LoginLog, String> {}
