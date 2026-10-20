package com.pei.clarity.repository;

import com.pei.clarity.model.entity.LoginLog;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface LoginLogRepository extends MongoRepository<LoginLog, String> {}
