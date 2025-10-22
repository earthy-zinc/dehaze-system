package com.pei.dehaze.database;

import com.mongodb.client.MongoCollection;
import jakarta.annotation.Resource;
import org.bson.Document;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.mongodb.core.MongoTemplate;

@SpringBootTest
public class MongodbTest {

    @Resource
    private MongoTemplate mongoTemplate;

    @Test
    public void test() {
        MongoCollection<Document> mongoCollection = mongoTemplate.createCollection("bl_comment");
    }
}
