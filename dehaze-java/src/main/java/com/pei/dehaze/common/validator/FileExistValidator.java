package com.pei.dehaze.common.validator;

import jakarta.validation.ConstraintValidator;
import jakarta.validation.ConstraintValidatorContext;

import java.io.File;

public class FileExistValidator implements ConstraintValidator<FileExists, String> {

    @Override
    public boolean isValid(String value, ConstraintValidatorContext context) {
        // null 或空字符串视为可选字段，校验通过
        if (value == null || value.isEmpty()) {
            return true;
        }
        File file = new File(value);
        return file.exists(); // 路径存在
    }
}
