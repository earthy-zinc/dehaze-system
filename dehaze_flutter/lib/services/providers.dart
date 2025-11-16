import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import '../core/network/dio_client.dart';
import '../core/network/network_info.dart';
import '../features/dehaze/data/datasources/dehaze_local_datasource.dart';
import '../features/dehaze/data/datasources/dehaze_remote_datasource.dart';
import '../features/dehaze/data/repositories/dehaze_repository_impl.dart';
import '../features/dehaze/domain/repositories/dehaze_repository.dart';

// 基础设施服务Providers
final sharedPreferencesProvider = Provider<SharedPreferences>((ref) {
  throw UnimplementedError('SharedPreferences must be initialized in main.dart');
});

final dioClientProvider = Provider<DioClient>((ref) {
  return DioClientImpl();
});

final networkInfoProvider = Provider<NetworkInfo>((ref) {
  return NetworkInfoImpl(Connectivity());
});

// 数据源Providers
final dehazeLocalDataSourceProvider = Provider<DehazeLocalDataSource>((ref) {
  return DehazeLocalDataSourceImpl(ref.read(sharedPreferencesProvider));
});

final dehazeRemoteDataSourceProvider = Provider<DehazeRemoteDataSource>((ref) {
  return DehazeRemoteDataSourceImpl(ref.read(dioClientProvider));
});

// 仓库Providers
final dehazeRepositoryProvider = Provider<DehazeRepository>((ref) {
  return DehazeRepositoryImpl(
    localDataSource: ref.read(dehazeLocalDataSourceProvider),
    remoteDataSource: ref.read(dehazeRemoteDataSourceProvider),
    networkInfo: ref.read(networkInfoProvider),
  );
});