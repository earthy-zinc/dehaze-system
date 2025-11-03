# 第6章：常用UI组件详解

## 📖 章节概述

本章将详细介绍Android开发中常用的UI组件，包括基础文本组件、交互组件、图像组件和高级列表组件。通过学习这些组件的使用方法、属性配置和最佳实践，您将能够构建功能丰富、用户友好的Android应用界面。

## 🎯 学习目标

- 掌握TextView文本组件的高级用法和样式设置
- 熟练使用Button、EditText等交互组件
- 了解ImageView图片处理和加载优化
- 掌握RecyclerView列表组件的使用
- 学会使用Material Design组件库
- 能够根据需求选择合适的UI组件

## 📝 文本组件详解

### TextView高级用法

TextView是Android中最常用的文本显示组件，支持丰富的文本格式和样式：

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp"
    tools:context=".TextViewExample">

    <!-- 基础TextView -->
    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="基础文本"
        android:textSize="16sp"
        android:textColor="@color/black"
        android:layout_marginBottom="16dp" />

    <!-- 带样式的TextView -->
    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="粗体文本"
        android:textSize="18sp"
        android:textStyle="bold"
        android:textColor="@color/primary"
        android:layout_marginBottom="16dp" />

    <!-- 带阴影的TextView -->
    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="带阴影的文本"
        android:textSize="20sp"
        android:textColor="@android:color/white"
        android:shadowColor="@android:color/black"
        android:shadowDx="2"
        android:shadowDy="2"
        android:shadowRadius="4"
        android:layout_marginBottom="16dp" />

    <!-- 多行文本 -->
    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="这是一个多行文本示例，当文本内容超过一行时，会自动换行显示。可以设置最大行数限制，以及省略号的显示方式。"
        android:textSize="14sp"
        android:textColor="@color/text_primary"
        android:maxLines="3"
        android:ellipsize="end"
        android:lineSpacingExtra="4dp"
        android:lineSpacingMultiplier="1.2"
        android:layout_marginBottom="16dp" />

    <!-- 带图标的TextView -->
    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="带图标的文本"
        android:textSize="16sp"
        android:drawableStart="@drawable/ic_star"
        android:drawablePadding="8dp"
        android:gravity="center_vertical"
        android:layout_marginBottom="16dp" />

    <!-- 可选中TextView -->
    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="可选择的文本，可以长按选择复制"
        android:textSize="14sp"
        android:textIsSelectable="true"
        android:background="?attr/selectableItemBackground"
        android:padding="12dp"
        android:layout_marginBottom="16dp" />

    <!-- 带边框和背景的TextView -->
    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="带边框的文本"
        android:textSize="16sp"
        android:textColor="@color/primary"
        android:gravity="center"
        android:padding="16dp"
        android:background="@drawable/text_border_background"
        android:layout_marginBottom="16dp" />

    <!-- 自动调整大小的TextView -->
    <TextView
        android:layout_width="match_parent"
        android:layout_height="60dp"
        android:text="自动调整大小的文本示例"
        android:gravity="center"
        android:background="@color/light_gray"
        android:autoSizeTextType="uniform"
        android:autoSizeMinTextSize="12sp"
        android:autoSizeMaxTextSize="24sp"
        android:autoSizeStepGranularity="2sp"
        android:layout_marginBottom="16dp" />

    <!-- 带链接的TextView -->
    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="访问官网：https://www.example.com 或发送邮件到：admin@example.com"
        android:textSize="14sp"
        android:autoLink="web|email"
        android:layout_marginBottom="16dp" />

    <!-- Material Design TextView -->
    <com.google.android.material.textview.MaterialTextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Material Design TextView"
        android:textSize="18sp"
        android:textAppearance="@style/TextAppearance.MaterialComponents.Headline6"
        android:layout_marginBottom="16dp" />

    <!-- 带文本样式的TextView -->
    <TextView
        android:id="@+id/styledTextView"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="样式文本示例"
        android:textSize="16sp"
        android:layout_marginBottom="16dp" />

</LinearLayout>
```

#### TextView动态样式设置

```java
/**
 * TextView高级用法示例
 */
public class TextViewExample extends AppCompatActivity {

    private TextView styledTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_text_view_example);

        styledTextView = findViewById(R.id.styledTextView);

        // 设置富文本
        setupRichText();

        // 设置文本样式
        setupTextStyle();

        // 设置文本动画
        setupTextAnimation();

        // 设置文本监听器
        setupTextListeners();
    }

    /**
     * 设置富文本
     */
    private void setupRichText() {
        String text = "这是富文本示例，包含不同颜色和样式的文字";
        SpannableStringBuilder spannable = new SpannableStringBuilder(text);

        // 设置部分文字颜色
        spannable.setSpan(
            new ForegroundColorSpan(Color.RED),
            2, 6,
            Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
        );

        // 设置部分文字背景色
        spannable.setSpan(
            new BackgroundColorSpan(Color.YELLOW),
            8, 12,
            Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
        );

        // 设置部分文字为粗体
        spannable.setSpan(
            new StyleSpan(Typeface.BOLD),
            14, 18,
            Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
        );

        // 设置部分文字为斜体
        spannable.setSpan(
            new StyleSpan(Typeface.ITALIC),
            20, 24,
            Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
        );

        // 设置下划线
        spannable.setSpan(
            new UnderlineSpan(),
            26, 30,
            Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
        );

        // 设置删除线
        spannable.setSpan(
            new StrikethroughSpan(),
            32, 36,
            Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
        );

        // 设置点击链接
        ClickableSpan clickableSpan = new ClickableSpan() {
            @Override
            public void onClick(@NonNull View widget) {
                Toast.makeText(TextViewExample.this, "点击了链接", Toast.LENGTH_SHORT).show();
            }
        };
        spannable.setSpan(clickableSpan, 38, 42, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);

        styledTextView.setText(spannable);
        styledTextView.setMovementMethod(LinkMovementMethod.getInstance());
    }

    /**
     * 设置文本样式
     */
    private void setupTextStyle() {
        // 设置字体
        Typeface typeface = Typeface.createFromAsset(getAssets(), "fonts/custom_font.ttf");
        styledTextView.setTypeface(typeface);

        // 设置文字阴影
        styledTextView.setShadowLayer(
            4f,  // 模糊半径
            2f,  // X偏移
            2f,  // Y偏移
            Color.BLACK  // 阴影颜色
        );

        // 设置字母间距
        styledTextView.setLetterSpacing(0.1f);

        // 设置行间距
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            styledTextView.setLineSpacing(4f, 1.2f);
        }
    }

    /**
     * 设置文本动画
     */
    private void setupTextAnimation() {
        // 文本淡入动画
        AlphaAnimation fadeIn = new AlphaAnimation(0f, 1f);
        fadeIn.setDuration(1000);
        fadeIn.setStartOffset(500);
        styledTextView.startAnimation(fadeIn);

        // 文本逐字显示效果
        animateText("这是逐字显示的文本效果示例");
    }

    /**
     * 文本逐字显示动画
     */
    private void animateText(String text) {
        final Handler handler = new Handler();
        final int[] index = {0};
        final long delay = 100; // 每个字符的显示间隔

        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                if (index[0] <= text.length()) {
                    styledTextView.setText(text.substring(0, index[0]));
                    index[0]++;
                    handler.postDelayed(this, delay);
                }
            }
        };

        handler.post(runnable);
    }

    /**
     * 设置文本监听器
     */
    private void setupTextListeners() {
        // 文本变化监听器
        styledTextView.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
                Log.d("TextView", "文本改变前: " + s);
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                Log.d("TextView", "文本改变中: " + s);
            }

            @Override
            public void afterTextChanged(Editable s) {
                Log.d("TextView", "文本改变后: " + s);
            }
        });
    }
}
```

### EditText输入组件

EditText是用户输入文本的核心组件：

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <!-- 基础EditText -->
    <com.google.android.material.textfield.TextInputLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginBottom="16dp"
        app:hintEnabled="true"
        app:hint="用户名">

        <com.google.android.material.textfield.TextInputEditText
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:inputType="text"
            android:maxLines="1" />

    </com.google.android.material.textfield.TextInputLayout>

    <!-- 密码输入框 -->
    <com.google.android.material.textfield.TextInputLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginBottom="16dp"
        app:hintEnabled="true"
        app:hint="密码"
        app:passwordToggleEnabled="true">

        <com.google.android.material.textfield.TextInputEditText
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:inputType="textPassword" />

    </com.google.android.material.textfield.TextInputLayout>

    <!-- 邮箱输入框 -->
    <com.google.android.material.textfield.TextInputLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginBottom="16dp"
        app:hintEnabled="true"
        app:hint="邮箱地址"
        app:helperText="请输入有效的邮箱地址"
        app:prefixText="@string/email_prefix"
        app:prefixTextColor="@color/primary">

        <com.google.android.material.textfield.TextInputEditText
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:inputType="textEmailAddress"
            android:maxLines="1" />

    </com.google.android.material.textfield.TextInputLayout>

    <!-- 数字输入框 -->
    <com.google.android.material.textfield.TextInputLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginBottom="16dp"
        app:hintEnabled="true"
        app:hint="年龄"
        app:suffixText="岁">

        <com.google.android.material.textfield.TextInputEditText
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:inputType="number"
            android:maxLines="1" />

    </com.google.android.material.textfield.TextInputLayout>

    <!-- 多行文本输入 -->
    <com.google.android.material.textfield.TextInputLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginBottom="16dp"
        app:hintEnabled="true"
        app:hint="个人简介"
        app:helperText="最多200个字符"
        app:counterEnabled="true"
        app:counterMaxLength="200">

        <com.google.android.material.textfield.TextInputEditText
            android:layout_width="match_parent"
            android:layout_height="120dp"
            android:gravity="top"
            android:inputType="textMultiLine|textCapSentences"
            android:maxLines="5" />

    </com.google.android.material.textfield.TextInputLayout>

    <!-- 带图标的输入框 -->
    <com.google.android.material.textfield.TextInputLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginBottom="16dp"
        app:hintEnabled="true"
        app:hint="搜索"
        app:startIconDrawable="@drawable/ic_search"
        app:endIconMode="clear_text">

        <com.google.android.material.textfield.TextInputEditText
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:inputType="text"
            android:maxLines="1" />

    </com.google.android.material.textfield.TextInputLayout>

    <!-- 错误状态输入框 -->
    <com.google.android.material.textfield.TextInputLayout
        android:id="@+id/errorInputLayout"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:layout_marginBottom="16dp"
        app:hintEnabled="true"
        app:hint="手机号码"
        app:errorEnabled="true">

        <com.google.android.material.textfield.TextInputEditText
            android:id="@+id/phoneEditText"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:inputType="phone"
            android:maxLines="1" />

    </com.google.android.material.textfield.TextInputLayout>

</LinearLayout>
```

#### EditText验证和格式化

```java
/**
 * EditText输入验证和格式化示例
 */
public class EditTextExample extends AppCompatActivity {

    private TextInputEditText phoneEditText;
    private TextInputLayout errorInputLayout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_edit_text_example);

        phoneEditText = findViewById(R.id.phoneEditText);
        errorInputLayout = findViewById(R.id.errorInputLayout);

        setupInputValidation();
        setupInputFormatting();
        setupInputFilters();
    }

    /**
     * 设置输入验证
     */
    private void setupInputValidation() {
        phoneEditText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {}

            @Override
            public void afterTextChanged(Editable s) {
                validatePhoneNumber(s.toString());
            }
        });
    }

    /**
     * 验证手机号码
     */
    private void validatePhoneNumber(String phone) {
        if (phone.isEmpty()) {
            errorInputLayout.setError(null);
        } else if (!isValidPhoneNumber(phone)) {
            errorInputLayout.setError("请输入有效的手机号码");
        } else {
            errorInputLayout.setError(null);
            // 可以添加成功状态的视觉反馈
            errorInputLayout.setHelperText("手机号码格式正确");
        }
    }

    /**
     * 检查手机号码是否有效
     */
    private boolean isValidPhoneNumber(String phone) {
        // 简单的手机号码验证正则表达式
        String phonePattern = "^1[3-9]\\d{9}$";
        return phone.matches(phonePattern);
    }

    /**
     * 设置输入格式化
     */
    private void setupInputFormatting() {
        phoneEditText.addTextChangedListener(new TextWatcher() {
            private boolean isFormatting = false;

            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {}

            @Override
            public void afterTextChanged(Editable s) {
                if (isFormatting) return;

                isFormatting = true;
                formatPhoneNumber(s);
                isFormatting = false;
            }
        });
    }

    /**
     * 格式化手机号码显示
     */
    private void formatPhoneNumber(Editable editable) {
        String phone = editable.toString().replaceAll("[^\\d]", ""); // 只保留数字

        if (phone.length() >= 7) {
            String formatted = String.format("%s-%s-%s",
                phone.substring(0, 3),
                phone.substring(3, 7),
                phone.substring(7)
            );
            editable.replace(0, editable.length(), formatted);
        } else if (phone.length() >= 3) {
            String formatted = String.format("%s-%s",
                phone.substring(0, 3),
                phone.substring(3)
            );
            editable.replace(0, editable.length(), formatted);
        }
    }

    /**
     * 设置输入过滤器
     */
    private void setupInputFilters() {
        // 限制输入长度
        InputFilter[] filters = new InputFilter[1];
        filters[0] = new InputFilter.LengthFilter(11); // 最多11位
        phoneEditText.setFilters(filters);

        // 自定义输入过滤器示例
        InputFilter alphaNumericFilter = new InputFilter() {
            @Override
            public CharSequence filter(CharSequence source, int start, int end,
                                       Spanned dest, int dstart, int dend) {
                // 只允许字母和数字
                for (int i = start; i < end; i++) {
                    if (!Character.isLetterOrDigit(source.charAt(i))) {
                        return "";
                    }
                }
                return null;
            }
        };
    }
}
```

## 🔘 交互组件详解

### Button按钮组件

按钮是用户与应用交互的主要方式：

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <!-- 基础Button -->
    <Button
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="基础按钮"
        android:layout_marginBottom="8dp" />

    <!-- 文本按钮 -->
    <com.google.android.material.button.MaterialButton
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="文本按钮"
        android:textAllCaps="false"
        style="@style/Widget.Material3.Button.TextButton"
        android:layout_marginBottom="8dp" />

    <!-- 轮廓按钮 -->
    <com.google.android.material.button.MaterialButton
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="轮廓按钮"
        style="@style/Widget.Material3.Button.OutlinedButton"
        android:layout_marginBottom="8dp" />

    <!-- 带图标的按钮 -->
    <com.google.android.material.button.MaterialButton
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="带图标的按钮"
        app:icon="@drawable/ic_download"
        app:iconGravity="textStart"
        android:layout_marginBottom="8dp" />

    <!-- 不同尺寸的按钮 -->
    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="horizontal"
        android:layout_marginBottom="16dp">

        <Button
            android:layout_width="0dp"
            android:layout_height="32dp"
            android:layout_weight="1"
            android:text="小按钮"
            android:textSize="12sp"
            android:layout_marginEnd="8dp" />

        <Button
            android:layout_width="0dp"
            android:layout_height="48dp"
            android:layout_weight="1"
            android:text="中按钮"
            android:layout_marginEnd="8dp" />

        <Button
            android:layout_width="0dp"
            android:layout_height="64dp"
            android:layout_weight="1"
            android:text="大按钮" />

    </LinearLayout>

    <!-- 带加载状态的按钮 -->
    <com.google.android.material.button.MaterialButton
        android:id="@+id/loadingButton"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="带加载状态的按钮"
        android:layout_marginBottom="8dp" />

    <!-- 切换按钮 -->
    <com.google.android.material.button.MaterialButtonToggleGroup
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="horizontal"
        android:layout_marginBottom="16dp">

        <com.google.android.material.button.MaterialButton
            android:layout_width="0dp"
            android:layout_height="wrap_content"
            android:layout_weight="1"
            android:text="选项1"
            style="@style/Widget.Material3.Button.OutlinedButton" />

        <com.google.android.material.button.MaterialButton
            android:layout_width="0dp"
            android:layout_height="wrap_content"
            android:layout_weight="1"
            android:text="选项2"
            style="@style/Widget.Material3.Button.OutlinedButton" />

        <com.google.android.material.button.MaterialButton
            android:layout_width="0dp"
            android:layout_height="wrap_content"
            android:layout_weight="1"
            android:text="选项3"
            style="@style/Widget.Material3.Button.OutlinedButton" />

    </com.google.android.material.button.MaterialButtonToggleGroup>

    <!-- 自定义背景按钮 -->
    <Button
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="渐变背景按钮"
        android:textColor="@android:color/white"
        android:background="@drawable/gradient_button_background"
        android:layout_marginBottom="8dp" />

    <!-- 圆形按钮 -->
    <com.google.android.material.button.MaterialButton
        android:layout_width="64dp"
        android:layout_height="64dp"
        android:insetTop="0dp"
        android:insetBottom="0dp"
        android:cornerRadius="32dp"
        app:icon="@drawable/ic_add"
        app:iconGravity="textStart"
        android:layout_gravity="center"
        android:layout_marginBottom="16dp" />

</LinearLayout>
```

#### Button交互效果和状态管理

```java
/**
 * Button交互示例
 */
public class ButtonExample extends AppCompatActivity {

    private MaterialButton loadingButton;
    private boolean isLoading = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_button_example);

        loadingButton = findViewById(R.id.loadingButton);

        setupButtonListeners();
        setupButtonStates();
    }

    /**
     * 设置按钮监听器
     */
    private void setupButtonListeners() {
        // 基础点击监听器
        View.OnClickListener clickListener = new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Button button = (Button) v;
                Toast.makeText(ButtonExample.this,
                    "点击了: " + button.getText(), Toast.LENGTH_SHORT).show();
            }
        };

        // 为所有按钮设置点击监听器
        for (int i = 0; i < ((ViewGroup) findViewById(R.id.buttonContainer)).getChildCount(); i++) {
            View child = ((ViewGroup) findViewById(R.id.buttonContainer)).getChildAt(i);
            if (child instanceof Button) {
                child.setOnClickListener(clickListener);
            }
        }

        // 加载按钮的特殊处理
        loadingButton.setOnClickListener(v -> {
            if (!isLoading) {
                startLoading();
            } else {
                stopLoading();
            }
        });

        // 长按监听器
        loadingButton.setOnLongClickListener(v -> {
            Toast.makeText(this, "长按按钮", Toast.LENGTH_SHORT).show();
            return true;
        });

        // 触摸监听器
        loadingButton.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        Log.d("Button", "按钮按下");
                        break;
                    case MotionEvent.ACTION_UP:
                        Log.d("Button", "按钮抬起");
                        break;
                }
                return false;
            }
        });
    }

    /**
     * 设置按钮状态
     */
    private void setupButtonStates() {
        // 禁用状态
        Button disabledButton = findViewById(R.id.disabledButton);
        disabledButton.setEnabled(false);

        // 选中状态
        MaterialButton selectedButton = findViewById(R.id.selectedButton);
        selectedButton.setSelected(true);

        // 按压状态动画
        loadingButton.setPressed(true);
        loadingButton.postDelayed(() -> loadingButton.setPressed(false), 100);
    }

    /**
     * 开始加载状态
     */
    private void startLoading() {
        isLoading = true;
        loadingButton.setEnabled(false);
        loadingButton.setText("加载中...");

        // 添加进度指示器
        ProgressBar progressBar = new ProgressBar(this);
        progressBar.setLayoutParams(new ViewGroup.LayoutParams(24, 24));

        SpannableStringBuilder builder = new SpannableStringBuilder("  加载中...");
        builder.setSpan(new ImageSpan(this, R.drawable.ic_loading), 0, 1,
            Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
        loadingButton.setText(builder);

        // 模拟加载过程
        new Handler().postDelayed(this::stopLoading, 3000);
    }

    /**
     * 停止加载状态
     */
    private void stopLoading() {
        isLoading = false;
        loadingButton.setEnabled(true);
        loadingButton.setText("带加载状态的按钮");
    }
}
```

### CheckBox和RadioButton

选择类组件提供多选和单选功能：

```xml
<?xml version="1.0" encoding="utf-8"?>
<ScrollView
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="16dp">

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical">

        <!-- CheckBox组 -->
        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="兴趣爱好（多选）"
            android:textSize="18sp"
            android:textStyle="bold"
            android:layout_marginBottom="16dp" />

        <com.google.android.material.checkbox.MaterialCheckBox
            android:id="@+id/readingCheckBox"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="阅读"
            android:layout_marginBottom="8dp" />

        <com.google.android.material.checkbox.MaterialCheckBox
            android:id="@+id/sportsCheckBox"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="运动"
            android:layout_marginBottom="8dp" />

        <com.google.android.material.checkbox.MaterialCheckBox
            android:id="@+id/musicCheckBox"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="音乐"
            android:layout_marginBottom="8dp" />

        <com.google.android.material.checkbox.MaterialCheckBox
            android:id="@+id/travelCheckBox"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="旅行"
            android:layout_marginBottom="8dp" />

        <!-- 自定义样式的CheckBox -->
        <com.google.android.material.checkbox.MaterialCheckBox
            android:id="@+id/customCheckBox"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="自定义样式"
            android:buttonTint="@color/primary"
            android:layout_marginBottom="24dp" />

        <!-- RadioGroup组 -->
        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="性别（单选）"
            android:textSize="18sp"
            android:textStyle="bold"
            android:layout_marginBottom="16dp" />

        <RadioGroup
            android:id="@+id/genderRadioGroup"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="vertical"
            android:layout_marginBottom="24dp">

            <com.google.android.material.radiobutton.MaterialRadioButton
                android:id="@+id/maleRadioButton"
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:text="男"
                android:layout_marginBottom="8dp" />

            <com.google.android.material.radiobutton.MaterialRadioButton
                android:id="@+id/femaleRadioButton"
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:text="女"
                android:layout_marginBottom="8dp" />

            <com.google.android.material.radiobutton.MaterialRadioButton
                android:id="@+id/otherRadioButton"
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:text="其他" />

        </RadioGroup>

        <!-- 水平RadioGroup -->
        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="年龄范围"
            android:textSize="18sp"
            android:textStyle="bold"
            android:layout_marginBottom="16dp" />

        <RadioGroup
            android:id="@+id/ageRadioGroup"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="24dp">

            <com.google.android.material.radiobutton.MaterialRadioButton
                android:id="@+id/age18RadioButton"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="18岁以下"
                android:layout_marginEnd="8dp" />

            <com.google.android.material.radiobutton.MaterialRadioButton
                android:id="@+id/age18to30RadioButton"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="18-30岁"
                android:layout_marginEnd="8dp" />

            <com.google.android.material.radiobutton.MaterialRadioButton
                android:id="@+id/age30RadioButton"
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="30岁以上" />

        </RadioGroup>

        <!-- 切换按钮 -->
        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="通知设置"
            android:textSize="18sp"
            android:textStyle="bold"
            android:layout_marginBottom="16dp" />

        <com.google.android.material.switchmaterial.SwitchMaterial
            android:id="@+id/notificationSwitch"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="推送通知"
            android:layout_marginBottom="8dp" />

        <com.google.android.material.switchmaterial.SwitchMaterial
            android:id="@+id/emailSwitch"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="邮件通知"
            android:checked="true"
            android:layout_marginBottom="8dp" />

        <com.google.android.material.switchmaterial.SwitchMaterial
            android:id="@+id/smsSwitch"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="短信通知"
            android:checked="false"
            android:layout_marginBottom="24dp" />

        <!-- 提交按钮 -->
        <Button
            android:id="@+id/submitButton"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="提交选择" />

        <!-- 显示结果 -->
        <TextView
            android:id="@+id/resultTextView"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:layout_marginTop="16dp"
            android:text="选择结果将显示在这里"
            android:textSize="14sp"
            android:textColor="@color/text_secondary" />

    </LinearLayout>

</ScrollView>
```

#### 选择组件事件处理

```java
/**
 * 选择组件事件处理示例
 */
public class SelectionExample extends AppCompatActivity {

    private MaterialCheckBox readingCheckBox, sportsCheckBox, musicCheckBox, travelCheckBox, customCheckBox;
    private RadioGroup genderRadioGroup, ageRadioGroup;
    private SwitchMaterial notificationSwitch, emailSwitch, smsSwitch;
    private TextView resultTextView;
    private Button submitButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_selection_example);

        initViews();
        setupListeners();
    }

    private void initViews() {
        // CheckBox
        readingCheckBox = findViewById(R.id.readingCheckBox);
        sportsCheckBox = findViewById(R.id.sportsCheckBox);
        musicCheckBox = findViewById(R.id.musicCheckBox);
        travelCheckBox = findViewById(R.id.travelCheckBox);
        customCheckBox = findViewById(R.id.customCheckBox);

        // RadioGroup
        genderRadioGroup = findViewById(R.id.genderRadioGroup);
        ageRadioGroup = findViewById(R.id.ageRadioGroup);

        // Switch
        notificationSwitch = findViewById(R.id.notificationSwitch);
        emailSwitch = findViewById(R.id.emailSwitch);
        smsSwitch = findViewById(R.id.smsSwitch);

        // 其他
        resultTextView = findViewById(R.id.resultTextView);
        submitButton = findViewById(R.id.submitButton);
    }

    private void setupListeners() {
        // CheckBox监听器
        CompoundButton.OnCheckedChangeListener checkBoxListener = (buttonView, isChecked) -> {
            String text = ((CheckBox) buttonView).getText().toString();
            Log.d("Selection", text + " " + (isChecked ? "选中" : "取消"));
            updateCustomCheckBoxState();
        };

        readingCheckBox.setOnCheckedChangeListener(checkBoxListener);
        sportsCheckBox.setOnCheckedChangeListener(checkBoxListener);
        musicCheckBox.setOnCheckedChangeListener(checkBoxListener);
        travelCheckBox.setOnCheckedChangeListener(checkBoxListener);

        // 自定义CheckBox特殊处理
        customCheckBox.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                // 选中自定义CheckBox时，选中所有其他选项
                selectAllInterests();
            } else {
                // 取消选中时，清除所有其他选项
                clearAllInterests();
            }
        });

        // RadioGroup监听器
        genderRadioGroup.setOnCheckedChangeListener((group, checkedId) -> {
            String gender = "";
            switch (checkedId) {
                case R.id.maleRadioButton:
                    gender = "男";
                    break;
                case R.id.femaleRadioButton:
                    gender = "女";
                    break;
                case R.id.otherRadioButton:
                    gender = "其他";
                    break;
            }
            Log.d("Selection", "性别选择: " + gender);
        });

        ageRadioGroup.setOnCheckedChangeListener((group, checkedId) -> {
            String ageRange = "";
            switch (checkedId) {
                case R.id.age18RadioButton:
                    ageRange = "18岁以下";
                    break;
                case R.id.age18to30RadioButton:
                    ageRange = "18-30岁";
                    break;
                case R.id.age30RadioButton:
                    ageRange = "30岁以上";
                    break;
            }
            Log.d("Selection", "年龄选择: " + ageRange);
        });

        // Switch监听器
        notificationSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            Log.d("Selection", "推送通知: " + (isChecked ? "开启" : "关闭"));
        });

        emailSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            Log.d("Selection", "邮件通知: " + (isChecked ? "开启" : "关闭"));
        });

        smsSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            Log.d("Selection", "短信通知: " + (isChecked ? "开启" : "关闭"));
        });

        // 提交按钮
        submitButton.setOnClickListener(v -> showSelectionResult());
    }

    /**
     * 更新自定义CheckBox状态
     */
    private void updateCustomCheckBoxState() {
        boolean allSelected = readingCheckBox.isChecked() &&
                            sportsCheckBox.isChecked() &&
                            musicCheckBox.isChecked() &&
                            travelCheckBox.isChecked();

        customCheckBox.setOnCheckedChangeListener(null); // 临时移除监听器避免递归
        customCheckBox.setChecked(allSelected);
        customCheckBox.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                selectAllInterests();
            } else {
                clearAllInterests();
            }
        });
    }

    /**
     * 选中所有兴趣
     */
    private void selectAllInterests() {
        readingCheckBox.setChecked(true);
        sportsCheckBox.setChecked(true);
        musicCheckBox.setChecked(true);
        travelCheckBox.setChecked(true);
    }

    /**
     * 清除所有兴趣选择
     */
    private void clearAllInterests() {
        readingCheckBox.setChecked(false);
        sportsCheckBox.setChecked(false);
        musicCheckBox.setChecked(false);
        travelCheckBox.setChecked(false);
    }

    /**
     * 显示选择结果
     */
    private void showSelectionResult() {
        StringBuilder result = new StringBuilder();
        result.append("选择结果：\n\n");

        // 兴趣爱好
        result.append("兴趣爱好：");
        List<String> interests = new ArrayList<>();
        if (readingCheckBox.isChecked()) interests.add("阅读");
        if (sportsCheckBox.isChecked()) interests.add("运动");
        if (musicCheckBox.isChecked()) interests.add("音乐");
        if (travelCheckBox.isChecked()) interests.add("旅行");

        if (interests.isEmpty()) {
            result.append("无");
        } else {
            result.append(TextUtils.join("、", interests));
        }
        result.append("\n");

        // 性别
        int genderId = genderRadioGroup.getCheckedRadioButtonId();
        String gender = "";
        switch (genderId) {
            case R.id.maleRadioButton:
                gender = "男";
                break;
            case R.id.femaleRadioButton:
                gender = "女";
                break;
            case R.id.otherRadioButton:
                gender = "其他";
                break;
        }
        result.append("性别：").append(gender.isEmpty() ? "未选择" : gender).append("\n");

        // 年龄
        int ageId = ageRadioGroup.getCheckedRadioButtonId();
        String ageRange = "";
        switch (ageId) {
            case R.id.age18RadioButton:
                ageRange = "18岁以下";
                break;
            case R.id.age18to30RadioButton:
                ageRange = "18-30岁";
                break;
            case R.id.age30RadioButton:
                ageRange = "30岁以上";
                break;
        }
        result.append("年龄：").append(ageRange.isEmpty() ? "未选择" : ageRange).append("\n");

        // 通知设置
        result.append("通知设置：");
        List<String> notifications = new ArrayList<>();
        if (notificationSwitch.isChecked()) notifications.add("推送");
        if (emailSwitch.isChecked()) notifications.add("邮件");
        if (smsSwitch.isChecked()) notifications.add("短信");

        if (notifications.isEmpty()) {
            result.append("全部关闭");
        } else {
            result.append(TextUtils.join("、", notifications));
        }

        resultTextView.setText(result.toString());
    }
}
```

## 🖼️ ImageView图片组件

ImageView是显示图片的核心组件：

```xml
<?xml version="1.0" encoding="utf-8"?>
<ScrollView
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:padding="16dp">

    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="vertical">

        <!-- 基础ImageView -->
        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="基础图片显示"
            android:textSize="16sp"
            android:textStyle="bold"
            android:layout_marginBottom="8dp" />

        <ImageView
            android:layout_width="match_parent"
            android:layout_height="200dp"
            android:src="@drawable/sample_image"
            android:scaleType="centerCrop"
            android:background="@drawable/image_border"
            android:layout_marginBottom="16dp" />

        <!-- 不同缩放类型 -->
        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="不同缩放类型"
            android:textSize="16sp"
            android:textStyle="bold"
            android:layout_marginBottom="8dp" />

        <!-- center -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="8dp">

            <TextView
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="center"
                android:gravity="center" />

            <ImageView
                android:layout_width="80dp"
                android:layout_height="80dp"
                android:src="@drawable/sample_image"
                android:scaleType="center"
                android:background="@color/light_gray" />

        </LinearLayout>

        <!-- centerCrop -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="8dp">

            <TextView
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="centerCrop"
                android:gravity="center" />

            <ImageView
                android:layout_width="80dp"
                android:layout_height="80dp"
                android:src="@drawable/sample_image"
                android:scaleType="centerCrop"
                android:background="@color/light_gray" />

        </LinearLayout>

        <!-- centerInside -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="8dp">

            <TextView
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="centerInside"
                android:gravity="center" />

            <ImageView
                android:layout_width="80dp"
                android:layout_height="80dp"
                android:src="@drawable/sample_image"
                android:scaleType="centerInside"
                android:background="@color/light_gray" />

        </LinearLayout>

        <!-- fitCenter -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="8dp">

            <TextView
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="fitCenter"
                android:gravity="center" />

            <ImageView
                android:layout_width="80dp"
                android:layout_height="80dp"
                android:src="@drawable/sample_image"
                android:scaleType="fitCenter"
                android:background="@color/light_gray" />

        </LinearLayout>

        <!-- fitXY -->
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="16dp">

            <TextView
                android:layout_width="0dp"
                android:layout_height="wrap_content"
                android:layout_weight="1"
                android:text="fitXY"
                android:gravity="center" />

            <ImageView
                android:layout_width="80dp"
                android:layout_height="80dp"
                android:src="@drawable/sample_image"
                android:scaleType="fitXY"
                android:background="@color/light_gray" />

        </LinearLayout>

        <!-- 圆形ImageView -->
        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="圆形图片"
            android:textSize="16sp"
            android:textStyle="bold"
            android:layout_marginBottom="8dp" />

        <ImageView
            android:layout_width="120dp"
            android:layout_height="120dp"
            android:src="@drawable/avatar_image"
            android:scaleType="centerCrop"
            android:background="@drawable/circle_background"
            android:layout_gravity="center"
            android:layout_marginBottom="16dp" />

        <!-- 带边框的ImageView -->
        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="带边框图片"
            android:textSize="16sp"
            android:textStyle="bold"
            android:layout_marginBottom="8dp" />

        <ImageView
            android:layout_width="match_parent"
            android:layout_height="150dp"
            android:src="@drawable/sample_image"
            android:scaleType="centerCrop"
            android:background="@drawable/image_with_border"
            android:padding="4dp"
            android:layout_marginBottom="16dp" />

        <!-- 可点击的ImageView -->
        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="可点击图片"
            android:textSize="16sp"
            android:textStyle="bold"
            android:layout_marginBottom="8dp" />

        <ImageView
            android:id="@+id/clickableImageView"
            android:layout_width="match_parent"
            android:layout_height="150dp"
            android:src="@drawable/sample_image"
            android:scaleType="centerCrop"
            android:background="?attr/selectableItemBackground"
            android:layout_marginBottom="16dp" />

        <!-- 占位符和错误图片 -->
        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="加载状态图片"
            android:textSize="16sp"
            android:textStyle="bold"
            android:layout_marginBottom="8dp" />

        <ImageView
            android:id="@+id/loadingImageView"
            android:layout_width="match_parent"
            android:layout_height="150dp"
            android:scaleType="centerCrop"
            android:background="@color/light_gray"
            android:layout_marginBottom="16dp" />

        <!-- 图片滤镜效果 -->
        <TextView
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:text="图片滤镜效果"
            android:textSize="16sp"
            android:textStyle="bold"
            android:layout_marginBottom="8dp" />

        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="horizontal"
            android:layout_marginBottom="16dp">

            <!-- 原图 -->
            <ImageView
                android:layout_width="0dp"
                android:layout_height="80dp"
                android:layout_weight="1"
                android:src="@drawable/sample_image"
                android:scaleType="centerCrop"
                android:layout_marginEnd="4dp" />

            <!-- 灰度滤镜 -->
            <ImageView
                android:layout_width="0dp"
                android:layout_height="80dp"
                android:layout_weight="1"
                android:src="@drawable/sample_image"
                android:scaleType="centerCrop"
                android:colorFilter="#80808080"
                android:layout_marginEnd="4dp" />

            <!-- 深色滤镜 -->
            <ImageView
                android:layout_width="0dp"
                android:layout_height="80dp"
                android:layout_weight="1"
                android:src="@drawable/sample_image"
                android:scaleType="centerCrop"
                android:colorFilter="#40000000" />

        </LinearLayout>

    </LinearLayout>

</ScrollView>
```

#### ImageView图片加载和处理

```java
/**
 * ImageView图片处理示例
 */
public class ImageViewExample extends AppCompatActivity {

    private ImageView clickableImageView;
    private ImageView loadingImageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_view_example);

        clickableImageView = findViewById(R.id.clickableImageView);
        loadingImageView = findViewById(R.id.loadingImageView);

        setupImageClickListeners();
        loadImageWithGlide();
        applyImageFilters();
        setupImageLoading();
    }

    /**
     * 设置图片点击监听器
     */
    private void setupImageClickListeners() {
        clickableImageView.setOnClickListener(v -> {
            // 点击图片时的动画效果
            animateImageClick(clickableImageView);

            Toast.makeText(this, "点击了图片", Toast.LENGTH_SHORT).show();
        });

        clickableImageView.setOnLongClickListener(v -> {
            // 长按图片显示菜单
            showImageMenu(v);
            return true;
        });
    }

    /**
     * 图片点击动画
     */
    private void animateImageClick(ImageView imageView) {
        // 缩放动画
        ObjectAnimator scaleX = ObjectAnimator.ofFloat(imageView, "scaleX", 1f, 0.95f, 1f);
        ObjectAnimator scaleY = ObjectAnimator.ofFloat(imageView, "scaleY", 1f, 0.95f, 1f);

        AnimatorSet animatorSet = new AnimatorSet();
        animatorSet.playTogether(scaleX, scaleY);
        animatorSet.setDuration(150);
        animatorSet.start();
    }

    /**
     * 显示图片菜单
     */
    private void showImageMenu(View anchor) {
        PopupMenu popup = new PopupMenu(this, anchor);
        popup.getMenuInflater().inflate(R.menu.image_menu, popup.getMenu());

        popup.setOnMenuItemClickListener(item -> {
            switch (item.getItemId()) {
                case R.id.action_save:
                    saveImage();
                    return true;
                case R.id.action_share:
                    shareImage();
                    return true;
                case R.id.action_details:
                    showImageDetails();
                    return true;
                default:
                    return false;
            }
        });

        popup.show();
    }

    /**
     * 使用Glide加载图片
     */
    private void loadImageWithGlide() {
        String imageUrl = "https://picsum.photos/400/200";

        Glide.with(this)
            .load(imageUrl)
            .placeholder(R.drawable.placeholder_image)  // 占位符
            .error(R.drawable.error_image)              // 错误图片
            .diskCacheStrategy(DiskCacheStrategy.ALL)   // 缓存策略
            .centerCrop()                                // 缩放类型
            .into(new CustomTarget<Drawable>() {
                @Override
                public void onResourceReady(@NonNull Drawable resource,
                                            @Nullable Transition<? super Drawable> transition) {
                    // 图片加载成功
                    loadingImageView.setImageDrawable(resource);

                    // 淡入动画
                    loadingImageView.setAlpha(0f);
                    loadingImageView.animate()
                        .alpha(1f)
                        .setDuration(300)
                        .start();
                }

                @Override
                public void onLoadCleared(@Nullable Drawable placeholder) {
                    // 图片加载清除
                    loadingImageView.setImageDrawable(placeholder);
                }

                @Override
                public void onLoadFailed(@Nullable Drawable errorDrawable) {
                    // 图片加载失败
                    loadingImageView.setImageDrawable(errorDrawable);
                }
            });
    }

    /**
     * 应用图片滤镜效果
     */
    private void applyImageFilters() {
        // 创建彩色矩阵
        ColorMatrix colorMatrix = new ColorMatrix();

        // 灰度效果
        colorMatrix.setSaturation(0f);
        ColorMatrixColorFilter grayFilter = new ColorMatrixColorFilter(colorMatrix);

        // 复古效果
        ColorMatrix sepiaMatrix = new ColorMatrix();
        sepiaMatrix.set(new float[] {
            0.393f, 0.769f, 0.189f, 0, 0,
            0.349f, 0.686f, 0.168f, 0, 0,
            0.272f, 0.534f, 0.131f, 0, 0,
            0,     0,     0,     1, 0
        });
        ColorMatrixColorFilter sepiaFilter = new ColorMatrixColorFilter(sepiaMatrix);

        // 应用滤镜到对应的ImageView
        // imageView.setColorFilter(grayFilter);
    }

    /**
     * 设置图片加载状态
     */
    private void setupImageLoading() {
        // 显示加载进度
        ProgressBar progressBar = new ProgressBar(this);
        // loadingImageView.addView(progressBar);

        // 模拟异步加载
        new Handler().postDelayed(() -> {
            // 加载完成，隐藏进度条
            if (progressBar.getParent() != null) {
                ((ViewGroup) progressBar.getParent()).removeView(progressBar);
            }
        }, 2000);
    }

    /**
     * 保存图片
     */
    private void saveImage() {
        // 实现图片保存逻辑
        Toast.makeText(this, "图片已保存", Toast.LENGTH_SHORT).show();
    }

    /**
     * 分享图片
     */
    private void shareImage() {
        // 实现图片分享逻辑
        Toast.makeText(this, "分享图片", Toast.LENGTH_SHORT).show();
    }

    /**
     * 显示图片详情
     */
    private void showImageDetails() {
        // 获取图片信息
        Drawable drawable = clickableImageView.getDrawable();
        int width = drawable.getIntrinsicWidth();
        int height = drawable.getIntrinsicHeight();

        String details = String.format("图片尺寸：%d x %d", width, height);
        Toast.makeText(this, details, Toast.LENGTH_LONG).show();
    }
}
```

## 📋 RecyclerView列表组件

RecyclerView是现代Android开发中最常用的列表组件：

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <!-- 工具栏 -->
    <com.google.android.material.appbar.MaterialToolbar
        android:id="@+id/toolbar"
        android:layout_width="0dp"
        android:layout_height="?attr/actionBarSize"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:title="RecyclerView示例" />

    <!-- RecyclerView -->
    <androidx.recyclerview.widget.RecyclerView
        android:id="@+id/recyclerView"
        android:layout_width="0dp"
        android:layout_height="0dp"
        android:padding="8dp"
        android:clipToPadding="false"
        app:layout_constraintTop_toBottomOf="@id/toolbar"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent" />

    <!-- 空状态视图 -->
    <LinearLayout
        android:id="@+id/emptyStateLayout"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:orientation="vertical"
        android:gravity="center"
        android:visibility="gone"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent">

        <ImageView
            android:layout_width="120dp"
            android:layout_height="120dp"
            android:src="@drawable/ic_empty_list"
            android:alpha="0.3"
            app:tint="@color/text_secondary" />

        <TextView
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:layout_marginTop="16dp"
            android:text="列表为空"
            android:textSize="18sp"
            android:textColor="@color/text_secondary"
            android:textStyle="bold" />

        <TextView
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:layout_marginTop="8dp"
            android:text="暂无数据显示"
            android:textSize="14sp"
            android:textColor="@color/text_hint" />

        <Button
            android:id="@+id/refreshButton"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:layout_marginTop="16dp"
            android:text="刷新数据"
            style="@style/Widget.Material3.Button.TextButton" />

    </LinearLayout>

    <!-- 加载状态 -->
    <ProgressBar
        android:id="@+id/progressBar"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:visibility="gone"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent" />

    <!-- 悬浮添加按钮 -->
    <com.google.android.material.floatingactionbutton.FloatingActionButton
        android:id="@+id/addButton"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_margin="16dp"
        android:src="@drawable/ic_add"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent" />

</androidx.constraintlayout.widget.ConstraintLayout>
```

#### RecyclerView适配器实现

```java
/**
 * RecyclerView适配器示例
 */
public class RecyclerExample extends AppCompatActivity {

    private RecyclerView recyclerView;
    private ProgressBar progressBar;
    private LinearLayout emptyStateLayout;
    private Button refreshButton;
    private FloatingActionButton addButton;

    private ItemAdapter adapter;
    private List<Item> itemList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recycler_example);

        initViews();
        setupRecyclerView();
        loadData();
        setupListeners();
    }

    private void initViews() {
        recyclerView = findViewById(R.id.recyclerView);
        progressBar = findViewById(R.id.progressBar);
        emptyStateLayout = findViewById(R.id.emptyStateLayout);
        refreshButton = findViewById(R.id.refreshButton);
        addButton = findViewById(R.id.addButton);
    }

    private void setupRecyclerView() {
        // 初始化数据
        itemList = new ArrayList<>();

        // 创建适配器
        adapter = new ItemAdapter(itemList, new ItemAdapter.OnItemClickListener() {
            @Override
            public void onItemClick(Item item) {
                Toast.makeText(RecyclerExample.this,
                    "点击了: " + item.getTitle(), Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onItemLongClick(Item item, int position) {
                showItemMenu(item, position);
            }
        });

        // 设置RecyclerView
        recyclerView.setAdapter(adapter);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        // 添加分割线
        recyclerView.addItemDecoration(new DividerItemDecoration(this,
            DividerItemDecoration.VERTICAL));

        // 添加滑动删除
        new ItemTouchHelper(new ItemTouchHelper.SimpleCallback(0,
            ItemTouchHelper.LEFT | ItemTouchHelper.RIGHT) {
            @Override
            public boolean onMove(@NonNull RecyclerView recyclerView,
                                  @NonNull RecyclerView.ViewHolder viewHolder,
                                  @NonNull RecyclerView.ViewHolder target) {
                return false;
            }

            @Override
            public void onSwiped(@NonNull RecyclerView.ViewHolder viewHolder, int direction) {
                int position = viewHolder.getAdapterPosition();
                deleteItem(position);
            }
        }).attachToRecyclerView(recyclerView);
    }

    private void loadData() {
        showLoading(true);

        // 模拟异步加载数据
        new Handler().postDelayed(() -> {
            // 生成示例数据
            itemList.clear();
            for (int i = 1; i <= 20; i++) {
                itemList.add(new Item(
                    "项目 " + i,
                    "这是第 " + i + " 个项目的描述信息",
                    i % 3 == 0 ? Item.Type.HEADER : Item.Type.NORMAL
                ));
            }

            showLoading(false);
            adapter.updateItems(itemList);
        }, 1500);
    }

    private void setupListeners() {
        refreshButton.setOnClickListener(v -> loadData());

        addButton.setOnClickListener(v -> {
            int newItemId = itemList.size() + 1;
            Item newItem = new Item(
                "新项目 " + newItemId,
                "这是新添加的项目",
                Item.Type.NORMAL
            );
            itemList.add(0, newItem);
            adapter.notifyItemInserted(0);
            recyclerView.scrollToPosition(0);
        });
    }

    private void deleteItem(int position) {
        Item deletedItem = itemList.get(position);
        itemList.remove(position);
        adapter.notifyItemRemoved(position);

        // 显示撤销选项
        Snackbar.make(recyclerView, "已删除: " + deletedItem.getTitle(),
            Snackbar.LENGTH_LONG)
            .setAction("撤销", v -> {
                itemList.add(position, deletedItem);
                adapter.notifyItemInserted(position);
            })
            .show();
    }

    private void showItemMenu(Item item, int position) {
        PopupMenu popup = new PopupMenu(this, recyclerView.getChildAt(position));
        popup.getMenuInflater().inflate(R.menu.item_menu, popup.getMenu());

        popup.setOnMenuItemClickListener(menuItem -> {
            switch (menuItem.getItemId()) {
                case R.id.action_edit:
                    editItem(item, position);
                    return true;
                case R.id.action_delete:
                    deleteItem(position);
                    return true;
                case R.id.action_duplicate:
                    duplicateItem(item, position);
                    return true;
                default:
                    return false;
            }
        });

        popup.show();
    }

    private void editItem(Item item, int position) {
        // 实现编辑功能
        Toast.makeText(this, "编辑: " + item.getTitle(), Toast.LENGTH_SHORT).show();
    }

    private void duplicateItem(Item item, int position) {
        Item duplicatedItem = new Item(
            item.getTitle() + " (副本)",
            item.getDescription(),
            Item.Type.NORMAL
        );
        itemList.add(position + 1, duplicatedItem);
        adapter.notifyItemInserted(position + 1);
    }

    private void showLoading(boolean loading) {
        progressBar.setVisibility(loading ? View.VISIBLE : View.GONE);
        recyclerView.setVisibility(loading ? View.GONE : View.VISIBLE);
        emptyStateLayout.setVisibility(View.GONE);

        if (!loading && itemList.isEmpty()) {
            recyclerView.setVisibility(View.GONE);
            emptyStateLayout.setVisibility(View.VISIBLE);
        }
    }

    /**
     * 数据模型类
     */
    public static class Item {
        public enum Type {
            HEADER, NORMAL
        }

        private String title;
        private String description;
        private Type type;

        public Item(String title, String description, Type type) {
            this.title = title;
            this.description = description;
            this.type = type;
        }

        // Getters
        public String getTitle() { return title; }
        public String getDescription() { return description; }
        public Type getType() { return type; }
    }

    /**
     * ViewHolder类
     */
    public static class ItemViewHolder extends RecyclerView.ViewHolder {
        TextView titleTextView;
        TextView descriptionTextView;
        View dividerView;

        public ItemViewHolder(@NonNull View itemView) {
            super(itemView);
            titleTextView = itemView.findViewById(R.id.titleTextView);
            descriptionTextView = itemView.findViewById(R.id.descriptionTextView);
            dividerView = itemView.findViewById(R.id.dividerView);
        }

        public void bind(Item item) {
            titleTextView.setText(item.getTitle());
            descriptionTextView.setText(item.getDescription());

            // 根据类型设置不同的样式
            if (item.getType() == Item.Type.HEADER) {
                titleTextView.setTextSize(18f);
                titleTextView.setTypeface(null, Typeface.BOLD);
                dividerView.setVisibility(View.VISIBLE);
            } else {
                titleTextView.setTextSize(16f);
                titleTextView.setTypeface(null, Typeface.NORMAL);
                dividerView.setVisibility(View.GONE);
            }
        }
    }

    /**
     * 适配器类
     */
    public static class ItemAdapter extends RecyclerView.Adapter<ItemViewHolder> {

        private List<Item> items;
        private OnItemClickListener listener;

        public interface OnItemClickListener {
            void onItemClick(Item item);
            void onItemLongClick(Item item, int position);
        }

        public ItemAdapter(List<Item> items, OnItemClickListener listener) {
            this.items = items;
            this.listener = listener;
        }

        @NonNull
        @Override
        public ItemViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_recycler, parent, false);
            return new ItemViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull ItemViewHolder holder, int position) {
            Item item = items.get(position);
            holder.bind(item);

            // 设置点击监听器
            holder.itemView.setOnClickListener(v -> {
                if (listener != null) {
                    listener.onItemClick(item);
                }
            });

            holder.itemView.setOnLongClickListener(v -> {
                if (listener != null) {
                    listener.onItemLongClick(item, holder.getAdapterPosition());
                }
                return true;
            });
        }

        @Override
        public int getItemCount() {
            return items.size();
        }

        public void updateItems(List<Item> newItems) {
            this.items.clear();
            this.items.addAll(newItems);
            notifyDataSetChanged();
        }
    }
}
```

## 🎯 小结

本章详细介绍了Android常用UI组件的使用方法和最佳实践，主要内容包括：

### 核心内容总结

1. **文本组件详解**
   - TextView的高级用法和样式设置
   - EditText输入验证和格式化
   - 富文本和SpannableString的使用
   - 文本动画和交互效果

2. **交互组件详解**
   - Button按钮组件的各种样式
   - 按钮状态管理和交互效果
   - CheckBox和RadioButton选择组件
   - SwitchMaterial开关组件

3. **图像组件详解**
   - ImageView的多种缩放类型
   - 图片加载和缓存策略
   - 图片滤镜和特效处理
   - 圆形图片和自定义样式

4. **列表组件详解**
   - RecyclerView的基本使用
   - 适配器模式和ViewHolder
   - 滑动删除和拖拽排序
   - 列表性能优化技巧

5. **Material Design组件**
   - MaterialButton、TextInputLayout等现代组件
   - 主题和样式系统
   - 响应式布局设计
   - 动画和过渡效果

### 学习要点

- **组件特性**：了解每种UI组件的特点和适用场景
- **属性配置**：熟练掌握组件的各种属性配置方法
- **事件处理**：理解用户交互事件的处理机制
- **性能优化**：了解UI组件的性能优化方法
- **Material Design**：掌握Material Design设计规范

### 下一步

下一章将学习Material Design设计规范，了解如何创建符合现代设计标准的用户界面。

## 📚 延伸阅读

- [Android Developers官方文档 - UI Components](https://developer.android.com/guide/topics/ui)
- [Material Design组件库](https://material.io/develop/android/docs/getting-started)
- [RecyclerView官方指南](https://developer.android.com/guide/topics/ui/layout/recyclerview)
- [Glide图片加载库](https://github.com/bumptech/glide)