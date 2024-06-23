package com.pei.dehaze.common.validator;

import jakarta.validation.ConstraintValidator;
import jakarta.validation.ConstraintValidatorContext;

import java.io.File;

public class DirExistValidator implements ConstraintValidator<DirExists, String> {

    @Override
    public boolean isValid(String value, ConstraintValidatorContext context) {
        File directory = new File(value);
        return directory.exists() && directory.isDirectory(); // 路径存在且为目录
    }
}
