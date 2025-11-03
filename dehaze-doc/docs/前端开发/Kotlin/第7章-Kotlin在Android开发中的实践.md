# 第7章：Kotlin在Android开发中的实践

## 📖 章节概述

Kotlin在2017年被Google宣布为Android官方开发语言，如今已成为Android开发的首选。本章将深入探讨Kotlin在Android开发中的实践应用，包括项目配置、架构设计、Jetpack组件使用、协程应用以及现代化的UI开发技巧。

**学习时长**: 约4-5天
**核心目标**: 掌握Kotlin在Android开发中的最佳实践，能够构建现代化、高质量的Android应用

---

## 7.1 Android项目Kotlin配置

### 7.1.1 项目级配置

```kotlin
// build.gradle.kts (Project level)
plugins {
    id("com.android.application") version "8.1.0" apply false
    id("com.android.library") version "8.1.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.0" apply false
    id("org.jetbrains.kotlin.kapt") version "1.9.0" apply false
    id("com.google.dagger.hilt.android") version "2.44" apply false
}

buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.google.gms:google-services:4.3.15")
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
        maven("https://jitpack.io")
    }
}
```

### 7.1.2 应用级配置

```kotlin
// build.gradle.kts (App level)
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.kapt")
    id("com.google.dagger.hilt.android")
    id("kotlin-parcelize")
    id("kotlin-kapt")
    id("com.google.gms.google-services")
}

android {
    namespace = "com.example.kotlinapp"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.kotlinapp"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
            isDebuggable = true
            applicationIdSuffix = ".debug"
        }

        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )

            // 签名配置
            signingConfig = signingConfigs.getByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
        freeCompilerArgs = listOf(
            "-Xjsr305=strict",
            "-opt-in=kotlin.RequiresOptIn",
            "-opt-in=kotlinx.coroutines.ExperimentalCoroutinesApi",
            "-opt-in=kotlinx.coroutines.FlowPreview"
        )
    }

    buildFeatures {
        viewBinding = true
        dataBinding = true
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.4"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    testOptions {
        unitTests {
            isIncludeAndroidResources = true
        }
    }
}

dependencies {
    // Kotlin标准库
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // Fragment & Activity
    implementation("androidx.fragment:fragment-ktx:1.6.2")
    implementation("androidx.activity:activity-ktx:1.8.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.2")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.6.2")
    implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.6.2")

    // Navigation Component
    implementation("androidx.navigation:navigation-fragment-ktx:2.7.6")
    implementation("androidx.navigation:navigation-ui-ktx:2.7.6")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // Room Database
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    kapt("androidx.room:room-compiler:2.6.1")

    // Retrofit & OkHttp
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

    // Dagger Hilt
    implementation("com.google.dagger:hilt-android:2.44")
    kapt("com.google.dagger:hilt-compiler:2.44")
    implementation("androidx.hilt:hilt-navigation-fragment:1.1.0")

    // Jetpack Compose
    implementation(platform("androidx.compose:compose-bom:2023.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.8.2")
    implementation("androidx.navigation:navigation-compose:2.7.6")
    implementation("androidx.hilt:hilt-navigation-compose:1.1.0")

    // Coil (Image Loading)
    implementation("io.coil-kt:coil-compose:2.5.0")

    // Testing
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.mockito:mockito-core:5.7.0")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation(platform("androidx.compose:compose-bom:2023.10.01"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}

// Hilt插件配置
kapt {
    correctErrorTypes = true
}
```

### 7.1.3 Kotlin编译器选项优化

```kotlin
// proguard-rules.pro - Kotlin特定的混淆规则

# Kotlin序列化相关
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt
-dontnote kotlinx.serialization.SerializationKt

-includeruntimeclass kotlin.Metadata
-includeruntimeclass kotlinx.serialization.internal.MetadataMapElement

-keep,includedescriptorclasses class com.example.kotlinapp.**$$serializer { *; }
-keepclassmembers class com.example.kotlinapp.** {
    *** Companion;
}
-keepclasseswithmembers class com.example.kotlinapp.** {
    kotlinx.serialization.KSerializer serializer(...);
}

# Coroutines相关
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembernames class kotlinx.** {
    volatile <fields>;
}

# ViewBinding相关
-keep class * extends androidx.viewbinding.ViewBinding {
    public static *** inflate(...);
    public static *** bind(...);
}

# Room相关
-keep class * extends androidx.room.RoomDatabase
-dontwarn androidx.room.paging.**

# DataBinding相关
-keep class androidx.databinding.** { *; }
-keep class * extends androidx.databinding.ViewDataBinding { *; }
```

---

## 7.2 ViewBinding与属性委托

### 7.2.1 ViewBinding基础使用

```kotlin
// MainActivity.kt - ViewBinding基础使用
class MainActivity : AppCompatActivity() {

    // 使用属性委托简化ViewBinding的初始化
    private val binding: ActivityMainBinding by viewBinding()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(binding.root)

        setupUI()
        setupListeners()
    }

    private fun setupUI() {
        binding.toolbar.title = "Kotlin App"
        setSupportActionBar(binding.toolbar)

        // 直接访问View，无需findViewById
        binding.welcomeText.text = "Welcome to Kotlin Android!"
        binding.welcomeText.setTextColor(Color.BLUE)

        // 使用扩展函数简化设置
        binding.loginButton.apply {
            text = "Login"
            setBackgroundColor(ContextCompat.getColor(this@MainActivity, R.color.primary))
            setOnClickListener { performLogin() }
        }

        binding.loadingProgressBar.visibility = View.GONE
    }

    private fun setupListeners() {
        binding.loginButton.setOnClickListener {
            validateAndLogin()
        }

        binding.forgotPasswordText.setOnClickListener {
            navigateToForgotPassword()
        }

        binding.signupLink.setOnClickListener {
            navigateToSignUp()
        }
    }

    private fun validateAndLogin() {
        val email = binding.emailInput.text.toString()
        val password = binding.passwordInput.text.toString()

        if (validateInput(email, password)) {
            showLoading(true)
            performLogin()
        }
    }

    private fun validateInput(email: String, password: String): Boolean {
        var isValid = true

        // 验证邮箱
        if (email.isBlank()) {
            binding.emailInput.error = "Email cannot be empty"
            isValid = false
        } else if (!isValidEmail(email)) {
            binding.emailInput.error = "Invalid email format"
            isValid = false
        } else {
            binding.emailInput.error = null
        }

        // 验证密码
        if (password.isBlank()) {
            binding.passwordInput.error = "Password cannot be empty"
            isValid = false
        } else if (password.length < 6) {
            binding.passwordInput.error = "Password must be at least 6 characters"
            isValid = false
        } else {
            binding.passwordInput.error = null
        }

        return isValid
    }

    private fun isValidEmail(email: String): Boolean {
        return Patterns.EMAIL_ADDRESS.matcher(email).matches()
    }

    private fun showLoading(show: Boolean) {
        binding.loadingProgressBar.visibility = if (show) View.VISIBLE else View.GONE
        binding.loginButton.isEnabled = !show
        binding.emailInput.isEnabled = !show
        binding.passwordInput.isEnabled = !show
    }

    private fun performLogin() {
        // 使用协程进行异步操作
        lifecycleScope.launch {
            try {
                val result = loginRepository.login(
                    binding.emailInput.text.toString(),
                    binding.passwordInput.text.toString()
                )

                handleLoginResult(result)
            } catch (e: Exception) {
                showError("Login failed: ${e.message}")
            } finally {
                showLoading(false)
            }
        }
    }

    private fun handleLoginResult(result: LoginResult) {
        if (result.success) {
            showToast("Login successful!")
            navigateToHome()
        } else {
            showError("Login failed: ${result.errorMessage}")
        }
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun showError(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG).show()
    }

    private fun navigateToHome() {
        // 导航逻辑
    }

    private fun navigateToForgotPassword() {
        // 忘记密码导航
    }

    private fun navigateToSignUp() {
        // 注册导航
    }
}
```

### 7.2.2 属性委托进阶

```kotlin
// FragmentPropertyDelegate.kt - Fragment属性委托

/**
 * 自动处理Fragment生命周期的属性委托
 * 避免内存泄漏和重复初始化
 */
class FragmentPropertyDelegate<T : Any>(
    private val initializer: () -> T,
    private val resetOnDestroy: Boolean = true
) : ReadWriteProperty<Fragment, T> {

    private var value: T? = null
    private var isInitialized = false

    @Suppress("UNCHECKED_CAST")
    override fun getValue(thisRef: Fragment, property: KProperty<*>): T {
        if (!isInitialized || value == null) {
            value = initializer()
            isInitialized = true
        }
        return value as T
    }

    override fun setValue(thisRef: Fragment, property: KProperty<*>, value: T) {
        this.value = value
        isInitialized = true
    }

    fun reset() {
        value = null
        isInitialized = false
    }
}

/**
 * 扩展函数：为Fragment创建属性委托
 */
fun <T : Any> Fragment.fragmentProperty(
    initializer: () -> T,
    resetOnDestroy: Boolean = true
): ReadWriteProperty<Fragment, T> = FragmentPropertyDelegate(initializer, resetOnDestroy)

/**
 * Activity属性委托
 */
class ActivityPropertyDelegate<T : Any>(
    private val initializer: () -> T
) : ReadWriteProperty<AppCompatActivity, T> {

    private var value: T? = null

    @Suppress("UNCHECKED_CAST")
    override fun getValue(thisRef: AppCompatActivity, property: KProperty<*>): T {
        if (value == null) {
            value = initializer()
        }
        return value as T
    }

    override fun setValue(thisRef: AppCompatActivity, property: KProperty<*>, value: T) {
        this.value = value
    }
}

fun <T : Any> AppCompatActivity.activityProperty(
    initializer: () -> T
): ReadWriteProperty<AppCompatActivity, T> = ActivityPropertyDelegate(initializer)

// BaseFragment.kt - 使用属性委托的基础Fragment
abstract class BaseFragment : Fragment() {

    // ViewModel属性委托
    protected inline fun <reified T : ViewModel> viewModel(): Lazy<T> {
        return viewModels()
    }

    // 属性委托示例
    protected val adapter: RecyclerView.Adapter<*> by fragmentProperty {
        createAdapter()
    }

    protected var loading: Boolean by Delegates.observable(false) { _, _, newValue ->
        onLoadingStateChanged(newValue)
    }

    // 抽象方法，子类实现
    protected abstract fun createAdapter(): RecyclerView.Adapter<*>
    protected abstract fun onLoadingStateChanged(isLoading: Boolean)

    override fun onDestroyView() {
        super.onDestroyView()
        // 清理资源
    }
}

// ProfileFragment.kt - 使用属性委托的Fragment示例
class ProfileFragment : BaseFragment() {

    private var _binding: FragmentProfileBinding? = null
    private val binding: FragmentProfileBinding get() = _binding!!

    // 使用属性委托管理ViewModel
    private val viewModel: ProfileViewModel by viewModel()

    // 使用属性委托管理Presenter
    private val presenter: ProfilePresenter by fragmentProperty {
        ProfilePresenter(requireContext())
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentProfileBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupUI()
        observeViewModel()
    }

    override fun createAdapter(): RecyclerView.Adapter<*> {
        return ProfileAdapter()
    }

    override fun onLoadingStateChanged(isLoading: Boolean) {
        binding.progressBar.visibility = if (isLoading) View.VISIBLE else View.GONE
        binding.content.visibility = if (isLoading) View.GONE else View.VISIBLE
    }

    private fun setupUI() {
        binding.refreshButton.setOnClickListener {
            loadUserProfile()
        }

        binding.editButton.setOnClickListener {
            navigateToEditProfile()
        }

        // 设置RecyclerView
        binding.recyclerView.adapter = adapter
        binding.recyclerView.layoutManager = LinearLayoutManager(requireContext())
    }

    private fun observeViewModel() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewModel.uiState.collect { state ->
                when (state) {
                    is ProfileUiState.Loading -> loading = true
                    is ProfileUiState.Success -> {
                        loading = false
                        showProfile(state.profile)
                    }
                    is ProfileUiState.Error -> {
                        loading = false
                        showError(state.message)
                    }
                }
            }
        }
    }

    private fun loadUserProfile() {
        viewModel.loadProfile()
    }

    private fun showProfile(profile: Profile) {
        binding.nameText.text = profile.name
        binding.emailText.text = profile.email
        binding.phoneText.text = profile.phone

        Glide.with(this)
            .load(profile.avatarUrl)
            .placeholder(R.drawable.avatar_placeholder)
            .error(R.drawable.avatar_error)
            .circleCrop()
            .into(binding.avatarImage)

        // 更新RecyclerView数据
        (adapter as ProfileAdapter).submitList(profile.recentActivities)
    }

    private fun showError(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG).show()
    }

    private fun navigateToEditProfile() {
        findNavController().navigate(R.id.action_profileFragment_to_editProfileFragment)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
```

### 7.2.3 高级属性委托模式

```kotlin
// AdvancedDelegates.kt - 高级属性委托

/**
 * 双向绑定的属性委托
 */
class TwoWayBindingDelegate<T>(
    private val view: TextView,
    private val converter: (String) -> T,
    private val reverseConverter: (T) -> String = { it.toString() }
) : ReadWriteProperty<Any, T> {

    private var value: T? = null

    init {
        view.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                value = s?.toString()?.let(converter)
            }
        })
    }

    override fun getValue(thisRef: Any, property: KProperty<*>): T {
        return value ?: view.text.toString().let(converter)
    }

    override fun setValue(thisRef: Any, property: KProperty<*>, value: T) {
        this.value = value
        view.text = reverseConverter(value)
    }
}

/**
 * 扩展函数：为TextView创建双向绑定
 */
fun <T> TextView.twoWayBinding(
    converter: (String) -> T,
    reverseConverter: (T) -> String = { it.toString() }
): ReadWriteProperty<Any, T> = TwoWayBindingDelegate(this, converter, reverseConverter)

/**
 * 防抖点击属性委托
 */
class DebouncedClickDelegate(
    private val clickListener: (View) -> Unit,
    private val debounceTime: Long = 300
) : ReadWriteProperty<View, Boolean> {

    private var isClickable = true
    private val handler = Handler(Looper.getMainLooper())

    override fun getValue(thisRef: View, property: KProperty<*>): Boolean {
        return isClickable
    }

    override fun setValue(thisRef: View, property: KProperty<*>, value: Boolean) {
        isClickable = value
        thisRef.isEnabled = value
    }

    fun onClick(view: View) {
        if (isClickable) {
            clickListener(view)
            isClickable = false
            handler.postDelayed({
                isClickable = true
            }, debounceTime)
        }
    }
}

/**
 * 验证属性委托
 */
class ValidationDelegate<T>(
    private val initialValue: T,
    private val validator: (T) -> Boolean,
    private val onValidationChanged: (Boolean) -> Unit
) : ReadWriteProperty<Any, T> {

    private var value: T = initialValue
    private var isValid: Boolean = false

    override fun getValue(thisRef: Any, property: KProperty<*>): T {
        return value
    }

    override fun setValue(thisRef: Any, property: KProperty<*>, value: T) {
        val oldValid = isValid
        this.value = value
        this.isValid = validator(value)

        if (oldValid != this.isValid) {
            onValidationChanged(this.isValid)
        }
    }

    fun isValid(): Boolean = isValid
}

/**
 * 资源属性委托
 */
class ResourceDelegate<T>(
    private val context: Context,
    private val resourceId: Int,
    private val resourceGetter: (Context, Int) -> T
) : ReadOnlyProperty<Any, T> {

    private var cachedValue: T? = null

    override fun getValue(thisRef: Any, property: KProperty<*>): T {
        return cachedValue ?: resourceGetter(context, resourceId).also {
            cachedValue = it
        }
    }
}

/**
 * 扩展函数：创建资源属性委托
 */
fun <T> Context.resource(
    resourceId: Int,
    resourceGetter: (Context, Int) -> T
): ReadOnlyProperty<Any, T> = ResourceDelegate(this, resourceId, resourceGetter)

// 使用示例
class LoginActivity : AppCompatActivity() {

    private lateinit var binding: ActivityLoginBinding

    // 双向绑定示例
    private var email: String by binding.emailInput.twoWayBinding(
        converter = { it },
        reverseConverter = { it }
    )

    private var password: String by binding.passwordInput.twoWayBinding(
        converter = { it },
        reverseConverter = { it }
    )

    // 验证委托示例
    private var emailValid: Boolean by ValidationDelegate(
        initialValue = false,
        validator = { email -> Patterns.EMAIL_ADDRESS.matcher(email).matches() }
    ) { isValid ->
        binding.emailInput.error = if (isValid) null else "Invalid email"
    }

    // 资源委托示例
    private val appName: String by resource(R.string.app_name) { ctx, id -> ctx.getString(id) }
    private val primaryColor: Int by resource(R.color.primary) { ctx, id -> ContextCompat.getColor(ctx, id) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLoginBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupDebouncedClick()
    }

    private fun setupDebouncedClick() {
        val debouncedClick = DebouncedClickDelegate({ view ->
            when (view.id) {
                R.id.login_button -> performLogin()
                R.id.forgot_password -> navigateToForgotPassword()
            }
        })

        binding.loginButton.setOnClickListener { debouncedClick.onClick(it) }
        binding.forgotPasswordText.setOnClickListener { debouncedClick.onClick(it) }
    }
}
```

---

## 7.3 ViewModel与LiveData的Kotlin优化

### 7.3.1 ViewModel的Kotlin最佳实践

```kotlin
// BaseViewModel.kt - 基础ViewModel
abstract class BaseViewModel : ViewModel() {

    private val _loading = MutableStateFlow(false)
    val loading: StateFlow<Boolean> = _loading.asStateFlow()

    private val _error = MutableSharedFlow<String>()
    val error: SharedFlow<String> = _error.asSharedFlow()

    private val _success = MutableSharedFlow<String>()
    val success: SharedFlow<String> = _success.asSharedFlow()

    protected fun setLoading(isLoading: Boolean) {
        _loading.value = isLoading
    }

    protected fun showError(message: String) {
        _error.tryEmit(message)
    }

    protected fun showSuccess(message: String) {
        _success.tryEmit(message)
    }

    // 协程作用域
    protected fun launchViewModelScope(
        onError: (String) -> Unit = { showError(it) },
        block: suspend CoroutineScope.() -> Unit
    ) {
        viewModelScope.launch {
            try {
                block()
            } catch (e: Exception) {
                onError(e.message ?: "Unknown error")
                Log.e("ViewModel", "Error in viewModelScope", e)
            }
        }
    }

    // 网络请求封装
    protected suspend fun <T> safeApiCall(
        apiCall: suspend () -> T
    ): Result<T> {
        return try {
            Result.success(apiCall())
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}

// UserViewModel.kt - 用户相关ViewModel
@HiltViewModel
class UserViewModel @Inject constructor(
    private val userRepository: UserRepository,
    private val sessionManager: SessionManager
) : BaseViewModel() {

    // 使用StateFlow管理UI状态
    private val _uiState = MutableStateFlow(UserUiState())
    val uiState: StateFlow<UserUiState> = _uiState.asStateFlow()

    // 使用SharedFlow处理一次性事件
    private val _navigationEvent = MutableSharedFlow<NavigationEvent>()
    val navigationEvent: SharedFlow<NavigationEvent> = _navigationEvent.asSharedFlow()

    init {
        loadUserProfile()
    }

    private fun loadUserProfile() {
        launchViewModelScope {
            setLoading(true)

            when (val result = safeApiCall { userRepository.getCurrentUser() }) {
                is Result.Success -> {
                    _uiState.update { currentState ->
                        currentState.copy(
                            isLoading = false,
                            user = result.data,
                            isAuthenticated = true
                        )
                    }
                }
                is Result.Failure -> {
                    _uiState.update { currentState ->
                        currentState.copy(
                            isLoading = false,
                            error = result.exception.message
                        )
                    }
                    showError("Failed to load user profile")
                }
            }
        }
    }

    fun refreshProfile() {
        loadUserProfile()
    }

    fun updateProfile(updateRequest: ProfileUpdateRequest) {
        launchViewModelScope {
            setLoading(true)

            when (val result = safeApiCall {
                userRepository.updateProfile(updateRequest)
            }) {
                is Result.Success -> {
                    _uiState.update { currentState ->
                        currentState.copy(
                            isLoading = false,
                            user = result.data
                        )
                    }
                    showSuccess("Profile updated successfully")
                }
                is Result.Failure -> {
                    _uiState.update { currentState ->
                        currentState.copy(
                            isLoading = false,
                            error = result.exception.message
                        )
                    }
                    showError("Failed to update profile")
                }
            }
        }
    }

    fun logout() {
        launchViewModelScope {
            sessionManager.logout()
            _uiState.update { currentState ->
                currentState.copy(
                    isAuthenticated = false,
                    user = null
                )
            }
            _navigationEvent.tryEmit(NavigationEvent.NavigateToLogin)
        }
    }

    // 处理用户操作
    fun onUserAction(action: UserAction) {
        when (action) {
            is UserAction.EditProfile -> {
                _navigationEvent.tryEmit(NavigationEvent.NavigateToEditProfile)
            }
            is UserAction.ChangePassword -> {
                _navigationEvent.tryEmit(NavigationEvent.NavigateToChangePassword)
            }
            is UserAction.ViewSettings -> {
                _navigationEvent.tryEmit(NavigationEvent.NavigateToSettings)
            }
            is UserAction.Refresh -> {
                refreshProfile()
            }
        }
    }
}

// UI状态数据类
data class UserUiState(
    val isLoading: Boolean = false,
    val user: User? = null,
    val isAuthenticated: Boolean = false,
    val error: String? = null
)

// 导航事件密封类
sealed class NavigationEvent {
    object NavigateToLogin : NavigationEvent()
    object NavigateToEditProfile : NavigationEvent()
    object NavigateToChangePassword : NavigationEvent()
    object NavigateToSettings : NavigationEvent()
}

// 用户操作密封类
sealed class UserAction {
    object EditProfile : UserAction()
    object ChangePassword : UserAction()
    object ViewSettings : UserAction()
    object Refresh : UserAction()
}
```

### 7.3.2 LiveData的高级用法

```kotlin
// LiveDataExtensions.kt - LiveData扩展函数

/**
 * 将Flow转换为LiveData
 */
fun <T> Flow<T>.asLiveData(
    context: CoroutineContext = EmptyCoroutineContext,
    timeoutInMs: Long = DEFAULT_TIMEOUT
): LiveData<T> = liveData(context, timeoutInMs) {
    collect { value -> emit(value) }
}

/**
 * 组合多个LiveData
 */
fun <T, K, R> LiveData<T>.combine(
    liveData: LiveData<K>,
    transform: (T?, K?) -> R
): LiveData<R> = MediatorLiveData<R>().apply {
    addSource(this@combine) { value ->
        this.value = transform(value, liveData.value)
    }
    addSource(liveData) { value ->
        this.value = transform(this@combine.value, value)
    }
}

/**
 * 条件性地观察LiveData
 */
fun <T> LiveData<T>.observeIf(
    lifecycleOwner: LifecycleOwner,
    predicate: () -> Boolean,
    observer: (T) -> Unit
) {
    if (predicate()) {
        observe(lifecycleOwner, observer)
    }
}

/**
 * 防抖观察LiveData
 */
fun <T> LiveData<T>.observeDebounced(
    lifecycleOwner: LifecycleOwner,
    timeoutMs: Long = 300L,
    observer: (T) -> Unit
) {
    val handler = Handler(Looper.getMainLooper())
    var runnable: Runnable? = null

    observe(lifecycleOwner) { value ->
        runnable?.let { handler.removeCallbacks(it) }
        runnable = Runnable { observer(value) }
        handler.postDelayed(runnable!!, timeoutMs)
    }
}

// ProfileViewModel.kt - 使用LiveData的ViewModel
class ProfileViewModel @Inject constructor(
    private val profileRepository: ProfileRepository
) : ViewModel() {

    // 用户信息的LiveData
    private val _userProfile = MutableLiveData<UserProfile>()
    val userProfile: LiveData<UserProfile> = _userProfile

    // 加载状态的LiveData
    private val _loadingState = MutableLiveData<LoadingState>()
    val loadingState: LiveData<LoadingState> = _loadingState

    // 错误信息的LiveData
    private val _errorMessage = MutableLiveData<String>()
    val errorMessage: LiveData<String> = _errorMessage

    // 组合LiveData：根据加载状态和用户信息显示UI状态
    val uiState: LiveData<ProfileUiState> = combine(_userProfile, _loadingState) { profile, loading ->
        when (loading) {
            LoadingState.LOADING -> ProfileUiState.Loading
            LoadingState.SUCCESS -> ProfileUiState.Success(profile ?: UserProfile())
            LoadingState.ERROR -> ProfileUiState.Error(_errorMessage.value ?: "Unknown error")
        }
    }

    // Transformations示例
    val userDisplayName: LiveData<String> = Transformations.map(_userProfile) { profile ->
        "${profile.firstName} ${profile.lastName}"
    }

    val userAvatar: LiveData<String> = Transformations.switchMap(_userProfile) { profile ->
        if (profile.avatarUrl.isNotEmpty()) {
            MutableLiveData(profile.avatarUrl)
        } else {
            getDefaultAvatar(profile.gender)
        }
    }

    private fun getDefaultAvatar(gender: String): LiveData<String> {
        return when (gender.lowercase()) {
            "male" -> MutableLiveData("default_male_avatar.png")
            "female" -> MutableLiveData("default_female_avatar.png")
            else -> MutableLiveData("default_avatar.png")
        }
    }

    // 使用MediatorLiveData组合多个数据源
    val completeUserProfile: LiveData<CompleteUserProfile> = MediatorLiveData<CompleteUserProfile>().apply {
        var profile: UserProfile? = null
        var preferences: UserPreferences? = null
        var statistics: UserStatistics? = null

        fun updateIfReady() {
            if (profile != null && preferences != null && statistics != null) {
                value = CompleteUserProfile(profile!!, preferences!!, statistics!!)
            }
        }

        addSource(_userProfile) { profile ->
            this.profile = profile
            updateIfReady()
        }

        addSource(profileRepository.userPreferences) { preferences ->
            this.preferences = preferences
            updateIfReady()
        }

        addSource(profileRepository.userStatistics) { statistics ->
            this.statistics = statistics
            updateIfReady()
        }
    }

    fun loadProfile(userId: String) {
        viewModelScope.launch {
            _loadingState.value = LoadingState.LOADING

            try {
                val profile = profileRepository.getProfile(userId)
                _userProfile.postValue(profile)
                _loadingState.postValue(LoadingState.SUCCESS)
            } catch (e: Exception) {
                _errorMessage.postValue("Failed to load profile: ${e.message}")
                _loadingState.postValue(LoadingState.ERROR)
            }
        }
    }

    fun updateProfile(profile: UserProfile) {
        viewModelScope.launch {
            try {
                _loadingState.value = LoadingState.LOADING
                profileRepository.updateProfile(profile)
                _userProfile.postValue(profile)
                _loadingState.value = LoadingState.SUCCESS
            } catch (e: Exception) {
                _errorMessage.postValue("Failed to update profile: ${e.message}")
                _loadingState.value = LoadingState.ERROR
            }
        }
    }

    fun refreshProfile() {
        _userProfile.value?.let { profile ->
            loadProfile(profile.id)
        }
    }

    // 清理资源
    override fun onCleared() {
        super.onCleared()
        // 清理资源
    }
}

// 数据类定义
data class UserProfile(
    val id: String,
    val firstName: String,
    val lastName: String,
    val email: String,
    val avatarUrl: String = "",
    val gender: String = ""
)

data class UserPreferences(
    val theme: String = "light",
    val language: String = "en",
    val notifications: Boolean = true
)

data class UserStatistics(
    val loginCount: Int = 0,
    val lastLogin: Long = System.currentTimeMillis(),
    val totalActivityTime: Long = 0
)

data class CompleteUserProfile(
    val profile: UserProfile,
    val preferences: UserPreferences,
    val statistics: UserStatistics
)

enum class LoadingState {
    LOADING, SUCCESS, ERROR
}

sealed class ProfileUiState {
    object Loading : ProfileUiState()
    data class Success(val profile: UserProfile) : ProfileUiState()
    data class Error(val message: String) : ProfileUiState()
}
```

### 7.3.3 Repository模式的Kotlin实现

```kotlin
// UserRepository.kt - Repository模式实现
interface UserRepository {
    suspend fun getUser(id: String): Result<User>
    suspend fun saveUser(user: User): Result<User>
    suspend fun deleteUser(id: String): Result<Unit>
    fun getUserStream(id: String): Flow<User?>
    fun getAllUsers(): Flow<List<User>>
}

@Singleton
class UserRepositoryImpl @Inject constructor(
    private val localDataSource: UserDao,
    private val remoteDataSource: UserApi,
    private val cache: UserCache,
    @IoDispatcher private val ioDispatcher: CoroutineDispatcher
) : UserRepository {

    override suspend fun getUser(id: String): Result<User> = withContext(ioDispatcher) {
        // 首先检查缓存
        cache.get(id)?.let { cachedUser ->
            return@withContext Result.success(cachedUser)
        }

        // 检查本地数据库
        localDao.getUserById(id)?.let { localUser ->
            cache.put(localUser)
            return@withContext Result.success(localUser)
        }

        // 从网络获取
        try {
            val remoteUser = remoteDataSource.getUser(id)

            // 保存到本地数据库
            localDao.insertUser(remoteUser)

            // 保存到缓存
            cache.put(remoteUser)

            Result.success(remoteUser)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun saveUser(user: User): Result<User> = withContext(ioDispatcher) {
        try {
            // 保存到远程
            val savedUser = remoteDataSource.saveUser(user)

            // 保存到本地
            localDao.insertUser(savedUser)

            // 更新缓存
            cache.put(savedUser)

            Result.success(savedUser)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun deleteUser(id: String): Result<Unit> = withContext(ioDispatcher) {
        try {
            // 从远程删除
            remoteDataSource.deleteUser(id)

            // 从本地删除
            localDao.deleteUserById(id)

            // 从缓存删除
            cache.remove(id)

            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override fun getUserStream(id: String): Flow<User?> = channelFlow {
        // 发送缓存中的数据
        cache.get(id)?.let { send(it) }

        // 监听本地数据库变化
        localDao.getUserByIdFlow(id).collect { localUser ->
            localUser?.let {
                send(it)
                cache.put(it)
            }
        }

        // 定期从网络同步
        while (true) {
            delay(SYNC_INTERVAL)
            try {
                val remoteUser = remoteDataSource.getUser(id)
                localDao.insertUser(remoteUser)
                cache.put(remoteUser)
                send(remoteUser)
            } catch (e: Exception) {
                // 忽略网络错误，继续使用本地数据
            }
        }
    }.flowOn(ioDispatcher)

    override fun getAllUsers(): Flow<List<User>> = channelFlow {
        // 发送本地数据
        send(localDao.getAllUsers())

        // 监听本地数据库变化
        localDao.getAllUsersFlow().collect { users ->
            send(users)
            // 更新缓存
            users.forEach { cache.put(it) }
        }

        // 定期从网络同步
        while (true) {
            delay(SYNC_INTERVAL)
            try {
                val remoteUsers = remoteDataSource.getAllUsers()
                localDao.insertUsers(remoteUsers)
                remoteUsers.forEach { cache.put(it) }
                send(remoteUsers)
            } catch (e: Exception) {
                // 忽略网络错误，继续使用本地数据
            }
        }
    }.flowOn(ioDispatcher)

    companion object {
        private const val SYNC_INTERVAL = 60_000L // 1分钟
    }
}

// Cache实现
class UserCache @Inject constructor() {
    private val cache = mutableMapOf<String, User>()
    private val maxSize = 100

    fun get(id: String): User? = cache[id]

    fun put(user: User) {
        if (cache.size >= maxSize) {
            // 简单的LRU：移除第一个元素
            cache.entries.firstOrNull()?.let {
                cache.remove(it.key)
            }
        }
        cache[user.id] = user
    }

    fun remove(id: String) {
        cache.remove(id)
    }

    fun clear() {
        cache.clear()
    }
}
```

---

## 7.4 协程在Android中的应用

### 7.4.1 协程作用域的最佳实践

```kotlin
// CoroutineScopeProviders.kt - 协程作用域提供者

/**
 * MainActivity的协程作用域
 */
class MainActivityScopeProvider @Inject constructor() {

    private val _coroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    val coroutineScope: CoroutineScope = _coroutineScope

    fun cancel() {
        _coroutineScope.cancel()
    }
}

/**
 * ViewModel的协程作用域
 */
class ViewModelScopeProvider(
    private val viewModelScope: CoroutineScope
) {
    val scope: CoroutineScope = viewModelScope
}

/**
 * Fragment的协程作用域
 */
class FragmentScopeProvider(
    private val fragment: Fragment
) {
    val scope: CoroutineScope = fragment.viewLifecycleOwner.lifecycleScope
}

// CoroutineUtils.kt - 协程工具类

/**
 * 安全地在主线程执行
 */
fun runOnMainThread(block: () -> Unit) {
    if (Looper.myLooper() == Looper.getMainLooper()) {
        block()
    } else {
        Handler(Looper.getMainLooper()).post(block)
    }
}

/**
 * 带重试机制的协程执行
 */
suspend fun <T> withRetry(
    times: Int = 3,
    initialDelayMs: Long = 1000,
    maxDelayMs: Long = 5000,
    factor: Double = 2.0,
    block: suspend () -> T
): T {
    var currentDelay = initialDelayMs
    repeat(times - 1) {
        try {
            return block()
        } catch (e: Exception) {
            delay(currentDelay)
            currentDelay = (currentDelay * factor).toLong().coerceAtMost(maxDelayMs)
        }
    }
    return block() // 最后一次尝试
}

/**
 * 带超时的协程执行
 */
suspend fun <T> withTimeoutOrFail(
    timeoutMs: Long,
    timeoutMessage: String = "Operation timed out",
    block: suspend () -> T
): T {
    return try {
        withTimeout(timeoutMs) {
            block()
        }
    } catch (e: TimeoutCancellationException) {
        throw TimeoutException(timeoutMessage, e)
    }
}
```

### 7.4.2 网络请求的协程实现

```kotlin
// NetworkService.kt - 网络服务实现
@Singleton
class NetworkService @Inject constructor(
    private val apiService: ApiService,
    private val networkManager: NetworkManager,
    @IoDispatcher private val ioDispatcher: CoroutineDispatcher
) {

    /**
     * 执行网络请求的通用方法
     */
    private suspend fun <T> executeRequest(
        request: suspend () -> T
    ): NetworkResult<T> {
        return try {
            // 检查网络连接
            if (!networkManager.isConnected()) {
                return NetworkResult.NoNetwork
            }

            // 执行请求
            val result = withTimeoutOrFail(30_000, "Network request timeout") {
                request()
            }

            NetworkResult.Success(result)
        } catch (e: HttpException) {
            val statusCode = e.code()
            val errorMessage = e.message()

            when (statusCode) {
                in 400..499 -> NetworkResult.ClientError(statusCode, errorMessage ?: "Client error")
                in 500..599 -> NetworkResult.ServerError(statusCode, errorMessage ?: "Server error")
                else -> NetworkResult.UnknownError(errorMessage ?: "Unknown error")
            }
        } catch (e: IOException) {
            NetworkResult.NetworkError(e.message ?: "Network error")
        } catch (e: Exception) {
            NetworkResult.UnknownError(e.message ?: "Unknown error")
        }
    }

    /**
     * 带重试的网络请求
     */
    private suspend fun <T> executeRequestWithRetry(
        request: suspend () -> T,
        retryCount: Int = 3
    ): NetworkResult<T> {
        return withRetry(retryCount) {
            when (val result = executeRequest(request)) {
                is NetworkResult.Success -> result
                is NetworkResult.NetworkError -> {
                    if (networkManager.isConnected()) {
                        throw Exception("Network error with connection available")
                    } else {
                        result
                    }
                }
                else -> throw Exception(result.message)
            }
        }
    }

    /**
     * 获取用户信息
     */
    suspend fun getUser(userId: String): NetworkResult<User> {
        return executeRequestWithRetry {
            apiService.getUser(userId)
        }
    }

    /**
     * 上传文件
     */
    suspend fun uploadFile(
        file: File,
        onProgress: (Float) -> Unit = {}
    ): NetworkResult<UploadResult> {
        return executeRequest {
            val requestBody = ProgressRequestBody(
                file.asRequestBody("multipart/form-data".toMediaType()),
                onProgress
            )

            val multipartBody = MultipartBody.Part.createFormData(
                "file",
                file.name,
                requestBody
            )

            apiService.uploadFile(multipartBody)
        }
    }

    /**
     * 批量请求
     */
    suspend fun <T> executeBatchRequests(
        requests: List<suspend () -> T>
    ): NetworkResult<List<T>> {
        return try {
            val results = withContext(ioDispatcher) {
                requests.map { request ->
                    async {
                        when (val result = executeRequest(request)) {
                            is NetworkResult.Success -> result.data
                            else -> throw Exception(result.message)
                        }
                    }
                }.awaitAll()
            }

            NetworkResult.Success(results)
        } catch (e: Exception) {
            NetworkResult.UnknownError("Batch request failed: ${e.message}")
        }
    }
}

// ProgressRequestBody.kt - 带进度的RequestBody
class ProgressRequestBody(
    private val requestBody: RequestBody,
    private val onProgress: (Float) -> Unit
) : RequestBody() {

    override fun contentType(): MediaType? = requestBody.contentType()

    override fun contentLength(): Long = requestBody.contentLength()

    override fun writeTo(sink: BufferedSink) {
        val progressSink = CountingSink(sink) { bytesWritten, totalBytes ->
            val progress = if (totalBytes > 0) bytesWritten.toFloat() / totalBytes else 0f
            onProgress(progress)
        }

        val bufferedSink = progressSink.buffer()
        requestBody.writeTo(bufferedSink)
        bufferedSink.flush()
    }

    inner class CountingSink(
        private val delegate: BufferedSink,
        private val onProgress: (Long, Long) -> Unit
    ) : ForwardingSink(delegate) {

        private var bytesWritten = 0L
        private var contentLength = 0L

        override fun write(source: Buffer, byteCount: Long) {
            super.write(source, byteCount)
            if (contentLength == 0L) {
                contentLength = contentLength()
            }
            bytesWritten += byteCount
            onProgress(bytesWritten, contentLength)
        }
    }
}

// NetworkResult.kt - 网络结果封装
sealed class NetworkResult<out T> {
    data class Success<out T>(val data: T) : NetworkResult<T>()
    object NoNetwork : NetworkResult<Nothing>()
    data class NetworkError(val message: String) : NetworkResult<Nothing>()
    data class ClientError(val code: Int, val message: String) : NetworkResult<Nothing>()
    data class ServerError(val code: Int, val message: String) : NetworkResult<Nothing>()
    data class UnknownError(val message: String) : NetworkResult<Nothing>()

    val message: String
        get() = when (this) {
            is Success -> "Success"
            is NoNetwork -> "No network connection"
            is NetworkError -> message
            is ClientError -> "Client error ($code): $message"
            is ServerError -> "Server error ($code): $message"
            is UnknownError -> "Unknown error: $message"
        }

    val isSuccess: Boolean
        get() = this is Success

    val isError: Boolean
        get() = !isSuccess
}
```

### 7.4.3 数据库操作的协程实现

```kotlin
// DatabaseService.kt - 数据库服务实现
@Singleton
class DatabaseService @Inject constructor(
    private val appDatabase: AppDatabase,
    @IoDispatcher private val ioDispatcher: CoroutineDispatcher
) {

    /**
     * 事务执行
     */
    suspend fun <T> executeInTransaction(
        block: suspend () -> T
    ): T = withContext(ioDispatcher) {
        appDatabase.withTransaction {
            block()
        }
    }

    /**
     * 批量插入
     */
    suspend fun <T> insertAll(
        entities: List<T>,
        tableName: String
    ): List<Long> = executeInTransaction {
        when (tableName) {
            "users" -> {
                @Suppress("UNCHECKED_CAST")
                val userDao = appDatabase.userDao()
                @Suppress("UNCHECKED_CAST")
                userDao.insertAll(entities as List<User>)
            }
            else -> throw IllegalArgumentException("Unknown table: $tableName")
        }
    }

    /**
     * 分页查询
     */
    suspend fun <T> getPagedData(
        tableName: String,
        page: Int,
        pageSize: Int
    ): List<T> = withContext(ioDispatcher) {
        when (tableName) {
            "users" -> {
                val userDao = appDatabase.userDao()
                val offset = page * pageSize
                @Suppress("UNCHECKED_CAST")
                userDao.getPagedUsers(offset, pageSize) as List<T>
            }
            else -> throw IllegalArgumentException("Unknown table: $tableName")
        }
    }

    /**
     * 流式查询
     */
    fun <T> observeData(
        tableName: String,
        query: String? = null
    ): Flow<List<T>> = channelFlow {
        when (tableName) {
            "users" -> {
                val userDao = appDatabase.userDao()
                @Suppress("UNCHECKED_CAST")
                userDao.observeAllUsers().collect { users ->
                    send(users as List<T>)
                }
            }
            else -> throw IllegalArgumentException("Unknown table: $tableName")
        }
    }.flowOn(ioDispatcher)

    /**
     * 缓存管理
     */
    suspend fun <T> cacheData(
        key: String,
        data: T,
        ttlMs: Long = 60_000 // 1分钟
    ) = withContext(ioDispatcher) {
        val cacheEntry = CacheEntry(
            key = key,
            data = Gson().toJson(data),
            timestamp = System.currentTimeMillis(),
            ttlMs = ttlMs
        )
        appDatabase.cacheDao().insertCache(cacheEntry)
    }

    suspend fun <T> getCachedData(
        key: String,
        clazz: Class<T>
    ): T? = withContext(ioDispatcher) {
        val cacheEntry = appDatabase.cacheDao().getCache(key)
        cacheEntry?.let { entry ->
            if (System.currentTimeMillis() - entry.timestamp < entry.ttlMs) {
                Gson().fromJson(entry.data, clazz)
            } else {
                null
            }
        }
    }
}

// CacheEntry.kt - 缓存实体
@Entity(tableName = "cache")
data class CacheEntry(
    @PrimaryKey val key: String,
    val data: String,
    val timestamp: Long,
    val ttlMs: Long
)

// CacheDao.kt - 缓存DAO
@Dao
interface CacheDao {
    @Query("SELECT * FROM cache WHERE key = :key")
    suspend fun getCache(key: String): CacheEntry?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertCache(cacheEntry: CacheEntry)

    @Query("DELETE FROM cache WHERE key = :key")
    suspend fun deleteCache(key: String)

    @Query("DELETE FROM cache WHERE timestamp < :expireTime")
    suspend fun cleanExpiredCache(expireTime: Long)
}

// 数据库使用示例
@HiltViewModel
class DataViewModel @Inject constructor(
    private val databaseService: DatabaseService,
    private val networkService: NetworkService
) : ViewModel() {

    private val _users = MutableStateFlow<List<User>>(emptyList())
    val users: StateFlow<List<User>> = _users.asStateFlow()

    private val _loading = MutableStateFlow(false)
    val loading: StateFlow<Boolean> = _loading.asStateFlow()

    init {
        loadUsersFromDatabase()
        observeDatabaseChanges()
    }

    private fun loadUsersFromDatabase() {
        viewModelScope.launch {
            _loading.value = true
            try {
                val users = databaseService.getPagedData("users", 0, 50)
                _users.value = users
            } catch (e: Exception) {
                Log.e("DataViewModel", "Failed to load users", e)
            } finally {
                _loading.value = false
            }
        }
    }

    private fun observeDatabaseChanges() {
        viewModelScope.launch {
            databaseService.observeData<User>("users")
                .collect { users ->
                    _users.value = users
                }
        }
    }

    fun refreshUsers() {
        viewModelScope.launch {
            when (val result = networkService.executeBatchRequests(
                listOf {
                    networkService.getUser("1")
                }
            )) {
                is NetworkResult.Success -> {
                    // 保存到数据库
                    databaseService.insertAll(result.data, "users")
                }
                else -> {
                    Log.e("DataViewModel", "Failed to refresh users: ${result.message}")
                }
            }
        }
    }

    fun insertUsers(users: List<User>) {
        viewModelScope.launch {
            try {
                databaseService.executeInTransaction {
                    databaseService.insertAll(users, "users")
                }
            } catch (e: Exception) {
                Log.e("DataViewModel", "Failed to insert users", e)
            }
        }
    }
}
```

---

## 7.5 Jetpack Compose基础

### 7.5.1 Compose基础组件

```kotlin
// BasicComposeComponents.kt - 基础Compose组件

@Composable
fun BasicComposeScreen(
    viewModel: BasicViewModel = hiltViewModel(),
    onNavigate: (String) -> Unit = {}
) {
    // 获取UI状态
    val uiState by viewModel.uiState.collectAsState()
    val scaffoldState = rememberScaffoldState()

    // Scaffold - 基本页面结构
    Scaffold(
        modifier = Modifier.fillMaxSize(),
        scaffoldState = scaffoldState,
        topBar = {
            TopAppBar(
                title = { Text("Jetpack Compose") },
                navigationIcon = {
                    IconButton(onClick = { /* 处理导航 */ }) {
                        Icon(Icons.Default.Menu, contentDescription = "Menu")
                    }
                },
                actions = {
                    IconButton(onClick = { /* 处理搜索 */ }) {
                        Icon(Icons.Default.Search, contentDescription = "Search")
                    }
                }
            )
        },
        bottomBar = {
            BottomNavigation {
                BottomNavigationItem(
                    icon = { Icon(Icons.Default.Home, contentDescription = "Home") },
                    label = { Text("Home") },
                    selected = uiState.currentTab == BottomTab.HOME,
                    onClick = { viewModel.onTabSelected(BottomTab.HOME) }
                )
                BottomNavigationItem(
                    icon = { Icon(Icons.Default.Favorite, contentDescription = "Favorites") },
                    label = { Text("Favorites") },
                    selected = uiState.currentTab == BottomTab.FAVORITES,
                    onClick = { viewModel.onTabSelected(BottomTab.FAVORITES) }
                )
                BottomNavigationItem(
                    icon = { Icon(Icons.Default.Person, contentDescription = "Profile") },
                    label = { Text("Profile") },
                    selected = uiState.currentTab == BottomTab.PROFILE,
                    onClick = { viewModel.onTabSelected(BottomTab.PROFILE) }
                )
            }
        },
        floatingActionButton = {
            FloatingActionButton(
                onClick = { /* 处理添加操作 */ }
            ) {
                Icon(Icons.Default.Add, contentDescription = "Add")
            }
        }
    ) { paddingValues ->
        // 主要内容区域
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            when (uiState.currentTab) {
                BottomTab.HOME -> HomeContent(uiState = uiState, onNavigate = onNavigate)
                BottomTab.FAVORITES -> FavoritesContent(uiState = uiState)
                BottomTab.PROFILE -> ProfileContent(uiState = uiState)
            }

            // 加载指示器
            if (uiState.isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.align(Alignment.Center)
                )
            }
        }
    }
}

@Composable
fun HomeContent(
    uiState: BasicUiState,
    onNavigate: (String) -> Unit
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            // 搜索框
            SearchBar(
                query = uiState.searchQuery,
                onQueryChange = { /* 处理搜索 */ },
                onSearch = { /* 执行搜索 */ },
                active = false,
                onActiveChange = { /* 处理激活状态 */ },
                modifier = Modifier.fillMaxWidth()
            ) {
                // 搜索建议内容
            }
        }

        item {
            // 卡片网格
            LazyRow(
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                contentPadding = PaddingValues(horizontal = 16.dp)
            ) {
                items(uiState.featuredItems) { item ->
                    FeatureCard(
                        item = item,
                        onClick = { onNavigate("detail/${item.id}") }
                    )
                }
            }
        }

        item {
            Text(
                text = "Categories",
                style = MaterialTheme.typography.h6,
                modifier = Modifier.padding(vertical = 8.dp)
            )
        }

        // 分类网格
        item {
            LazyVerticalGrid(
                columns = GridCells.Fixed(2),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                items(uiState.categories) { category ->
                    CategoryCard(
                        category = category,
                        onClick = { onNavigate("category/${category.id}") }
                    )
                }
            }
        }
    }
}

@Composable
fun FeatureCard(
    item: FeatureItem,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .width(200.dp)
            .clickable { onClick() },
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            AsyncImage(
                model = item.imageUrl,
                contentDescription = item.title,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(120.dp),
                contentScale = ContentScale.Crop,
                placeholder = painterResource(R.drawable.placeholder)
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = item.title,
                style = MaterialTheme.typography.titleMedium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )

            Spacer(modifier = Modifier.height(4.dp))

            Text(
                text = item.description,
                style = MaterialTheme.typography.bodySmall,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun CategoryCard(
    category: Category,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = category.icon,
                contentDescription = category.name,
                modifier = Modifier.size(48.dp),
                tint = MaterialTheme.colorScheme.primary
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = category.name,
                style = MaterialTheme.typography.titleSmall,
                textAlign = TextAlign.Center
            )
        }
    }
}
```

### 7.5.2 状态管理与动画

```kotlin
// StateManagement.kt - 状态管理示例

@Composable
fun StateManagementScreen() {
    // remember - 记住状态
    var counter by remember { mutableStateOf(0) }

    // rememberSaveable - 记住状态并保存到Bundle
    var text by rememberSaveable { mutableStateOf("") }

    // derivedStateOf - 派生状态
    val isTextValid by remember {
        derivedStateOf { text.isNotBlank() && text.length >= 3 }
    }

    // collectAsState - 收集Flow状态
    val viewModel: StateViewModel = hiltViewModel()
    val items by viewModel.items.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // 状态展示
        Text("Counter: $counter")

        // 状态修改
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(onClick = { counter-- }) {
                Text("Decrease")
            }
            Button(onClick = { counter++ }) {
                Text("Increase")
            }
        }

        // 文本输入与验证
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text("Enter text") },
            isError = text.isNotBlank() && !isTextValid,
            supportingText = {
                if (text.isNotBlank() && !isTextValid) {
                    Text("Text must be at least 3 characters")
                }
            }
        )

        // 派生状态的使用
        Button(
            onClick = { /* 处理提交 */ },
            enabled = isTextValid
        ) {
            Text("Submit")
        }

        // Flow状态的展示
        LazyColumn {
            items(items) { item ->
                ItemRow(item = item)
            }
        }
    }
}

// AnimationExamples.kt - 动画示例

@Composable
fun AnimationExamples() {
    // AnimatedVisibility - 可见性动画
    var visible by remember { mutableStateOf(true) }

    Column(
        modifier = Modifier.padding(16.dp)
    ) {
        Button(
            onClick = { visible = !visible }
        ) {
            Text(if (visible) "Hide" else "Show")
        }

        AnimatedVisibility(
            visible = visible,
            enter = fadeIn() + expandVertically(),
            exit = fadeOut() + shrinkVertically()
        ) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 16.dp)
            ) {
                Text(
                    text = "This content animates in and out!",
                    modifier = Modifier.padding(16.dp)
                )
            }
        }

        // AnimatedContent - 内容切换动画
        var contentType by remember { mutableStateOf(ContentType.TEXT) }

        Button(
            onClick = {
                contentType = when (contentType) {
                    ContentType.TEXT -> ContentType.IMAGE
                    ContentType.IMAGE -> ContentType.TEXT
                }
            },
            modifier = Modifier.padding(vertical = 8.dp)
        ) {
            Text("Switch Content")
        }

        AnimatedContent(
            targetState = contentType,
            transitionSpec = { _, _ ->
                fadeIn() with fadeOut()
            }
        ) { type ->
            when (type) {
                ContentType.TEXT -> {
                    Text(
                        text = "This is text content",
                        modifier = Modifier.padding(16.dp)
                    )
                }
                ContentType.IMAGE -> {
                    Image(
                        painter = painterResource(R.drawable.sample_image),
                        contentDescription = "Sample image",
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp)
                    )
                }
            }
        }

        // rememberInfiniteTransition - 无限动画
        val infiniteTransition = rememberInfiniteTransition()
        val alpha by infiniteTransition.animateFloat(
            initialValue = 0f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(
                animation = tween(1000),
                repeatMode = RepeatMode.Reverse
            )
        )

        Box(
            modifier = Modifier
                .size(100.dp)
                .background(
                    Color.Blue,
                    shape = CircleShape
                ),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "Fading",
                color = Color.White,
                alpha = alpha
            )
        }
    }
}

// GestureHandling.kt - 手势处理

@Composable
fun GestureHandling() {
    var offsetX by remember { mutableStateOf(0f) }
    var offsetY by remember { mutableStateOf(0f) }

    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .size(100.dp)
                .background(
                    Color.Red,
                    shape = RoundedCornerShape(8.dp)
                )
                .offset { IntOffset(offsetX.toInt(), offsetY.toInt()) }
                .pointerInput(Unit) {
                    detectDragGestures { change ->
                        val dragAmount = change.position
                        offsetX = dragAmount.x
                        offsetY = dragAmount.y
                    }
                },
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "Drag me!",
                color = Color.White
            )
        }
    }
}

// CustomComposables.kt - 自定义组件

@Composable
fun LoadingButton(
    text: String,
    isLoading: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    loadingIndicatorColor: Color = MaterialTheme.colorScheme.onPrimary,
    textStyle: TextStyle = MaterialTheme.typography.labelLarge
) {
    Button(
        onClick = onClick,
        enabled = enabled && !isLoading,
        modifier = modifier,
        contentPadding = if (isLoading) {
            PaddingValues(16.dp)
        } else {
            ButtonDefaults.ContentPadding
        }
    ) {
        if (isLoading) {
            CircularProgressIndicator(
                modifier = Modifier.size(20.dp),
                color = loadingIndicatorColor,
                strokeWidth = 2.dp
            )
        } else {
            Text(
                text = text,
                style = textStyle
            )
        }
    }
}

@Composable
fun ExpandableCard(
    title: String,
    initiallyExpanded: Boolean = false,
    content: @Composable () -> Unit
) {
    var expanded by remember { mutableStateOf(initiallyExpanded) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium
                )

                IconButton(
                    onClick = { expanded = !expanded }
                ) {
                    Icon(
                        imageVector = if (expanded) {
                            Icons.Default.ExpandLess
                        } else {
                            Icons.Default.ExpandMore
                        },
                        contentDescription = if (expanded) "Collapse" else "Expand"
                    )
                }
            }

            AnimatedVisibility(
                visible = expanded,
                enter = expandVertically() + fadeIn(),
                exit = shrinkVertically() + fadeOut()
            ) {
                content()
            }
        }
    }
}
```

---

## 7.6 本章小结

### ✅ 核心概念掌握

通过本章学习，您已经掌握了Kotlin在Android开发中的全面应用：

1. **Android项目Kotlin配置**
   - Gradle配置的最佳实践
   - Kotlin编译器选项优化
   - 混淆规则的配置
   - 依赖管理的策略

2. **ViewBinding与属性委托**
   - ViewBinding的基础使用
   - 高级属性委托模式
   - Fragment生命周期的处理
   - 双向绑定和验证委托

3. **ViewModel与LiveData优化**
   - ViewModel的Kotlin最佳实践
   - LiveData的高级用法
   - Repository模式的实现
   - 状态管理的优化

4. **协程在Android中的应用**
   - 协程作用域的最佳实践
   - 网络请求的协程实现
   - 数据库操作的协程处理
   - 错误处理和重试机制

5. **Jetpack Compose基础**
   - 基础组件的使用
   - 状态管理和动画
   - 手势处理和自定义组件
   - 现代UI开发模式

### ✅ Android开发优势

| 特性 | Java开发 | Kotlin开发 | 优势程度 |
|------|----------|------------|----------|
| 代码简洁性 | 冗长 | 简洁 | ⭐⭐⭐⭐⭐ |
| 空安全 | 需要手动检查 | 内置空安全 | ⭐⭐⭐⭐⭐ |
| 协程支持 | 复杂的线程管理 | 简单的协程 | ⭐⭐⭐⭐⭐ |
| 扩展函数 | 工具类 | 原生支持 | ⭐⭐⭐⭐ |
| 数据类 | 大量样板代码 | 自动生成 | ⭐⭐⭐⭐⭐ |
| 互操作性 | 无原生态支持 | 完美互操作 | ⭐⭐⭐⭐ |

### ✅ 实战要点

1. **项目架构设计**
   - 使用MVVM架构模式
   - 合理分层和职责分离
   - 依赖注入的使用
   - 数据流的设计

2. **性能优化**
   - 协程的合理使用
   - 内存泄漏的避免
   - ViewBinding的正确使用
   - 图片加载和缓存

3. **用户体验**
   - 流畅的动画效果
   - 响应式的UI设计
   - 错误处理的友好提示
   - 无障碍功能的支持

### 📚 下一步学习

下一章我们将探索**Kotlin后端开发实战**，包括：
- Spring Boot + Kotlin的配置
- Ktor框架的使用
- 数据库操作和持久化
- RESTful API开发
- 微服务架构实践

这将帮助您在后端开发中充分发挥Kotlin的优势！

---

## 📝 章节练习

### 基础练习
1. 创建一个完整的Android应用：
   - 实现MVVM架构
   - 使用ViewBinding和协程
   - 包含网络请求和本地存储
   - 添加Jetpack Compose界面

2. 重构一个Java Android应用为Kotlin：
   - 转换Activity和Fragment
   - 使用Kotlin的扩展函数
   - 实现协程替代AsyncTask
   - 优化代码结构和性能

### 进阶练习
1. 实现一个完整的Jetpack Compose应用：
   - 包含多个屏幕和导航
   - 实现复杂的状态管理
   - 添加动画和手势处理
   - 集成网络和数据库

2. 创建一个Android架构组件库：
   - 封装常用的BaseActivity/BaseFragment
   - 实现通用的ViewModel和Repository
   - 提供网络和数据库的抽象层
   - 支持协程和Flow的使用

### 挑战练习
1. 构建一个模块化的Android应用：
   - 实现feature模块化架构
   - 支持动态加载功能
   - 提供统一的依赖注入配置
   - 实现跨模块的数据共享

2. 设计一个Android性能监控工具：
   - 监控应用性能指标
   - 提供实时性能报告
   - 支持自定义监控规则
   - 实现性能优化建议

---

**恭喜完成Kotlin在Android开发中的实践学习！您现在已经掌握了现代Android开发的所有核心技术，能够构建高质量、高性能的Android应用程序了！**