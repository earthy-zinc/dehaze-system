import 'package:flutter_riverpod/flutter_riverpod.dart';

class HomeState {

  const HomeState({this.isLoading = false, this.errorMessage});
  final bool isLoading;
  final String? errorMessage;

  HomeState copyWith({bool? isLoading, String? errorMessage}) => HomeState(
      isLoading: isLoading ?? this.isLoading,
      errorMessage: errorMessage,
    );
}

class HomeNotifier extends StateNotifier<HomeState> {
  HomeNotifier() : super(const HomeState());

  Future<void> initialize() async {
    state = state.copyWith(isLoading: true);
    try {
      // TODO(home): Load home page data
      await Future<void>.delayed(const Duration(seconds: 1)); // 模拟网络请求
      state = state.copyWith(isLoading: false);
    } on Exception catch (e) {
      state = state.copyWith(isLoading: false, errorMessage: e.toString());
    }
  }

  void clearError() {
    state = state.copyWith();
  }
}

final homeProvider = StateNotifierProvider<HomeNotifier, HomeState>(
  (ref) => HomeNotifier(),
);
