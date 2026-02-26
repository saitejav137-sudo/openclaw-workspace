import 'package:flutter/material.dart';
import '../services/api_service.dart';

class ConfigProvider extends ChangeNotifier {
  final ApiService _api = ApiService();

  bool _isLoading = false;
  Map<String, dynamic> _config = {};
  Map<String, dynamic> _stats = {};

  bool get isLoading => _isLoading;
  Map<String, dynamic> get config => _config;
  Map<String, dynamic> get stats => _stats;

  String get mode => _config['mode'] ?? 'ocr';
  bool get polling => _config['polling'] ?? false;
  double get pollInterval => (_config['poll_interval'] ?? 0.5).toDouble();
  String get action => _config['action'] ?? 'alt+o';
  double get actionDelay => (_config['action_delay'] ?? 1.5).toDouble();

  Future<void> loadConfig() async {
    _isLoading = true;
    notifyListeners();

    try {
      _config = await _api.getConfig();
    } catch (e) {
      _config = {};
    }

    _isLoading = false;
    notifyListeners();
  }

  Future<void> loadStats() async {
    try {
      _stats = await _api.getStats();
    } catch (e) {
      _stats = {};
    }
    notifyListeners();
  }

  Future<bool> updateConfig(Map<String, dynamic> newConfig) async {
    _isLoading = true;
    notifyListeners();

    final success = await _api.updateConfig(newConfig);

    _isLoading = false;
    notifyListeners();

    if (success) {
      _config = {..._config, ...newConfig};
    }

    return success;
  }

  Future<bool> setMode(String mode) async {
    return updateConfig({'mode': mode});
  }

  Future<bool> setPolling(bool enabled) async {
    return updateConfig({'polling': enabled});
  }

  Future<bool> setPollInterval(double interval) async {
    return updateConfig({'poll_interval': interval});
  }

  Future<bool> setAction(String action) async {
    return updateConfig({'action': action});
  }

  Future<bool> setActionDelay(double delay) async {
    return updateConfig({'action_delay': delay});
  }

  Future<bool> setTargetText(String text) async {
    return updateConfig({'target_text': text});
  }

  Future<bool> setRegion(List<int>? region) async {
    return updateConfig({'region': region});
  }

  Future<bool> setTargetColor(List<int>? color) async {
    return updateConfig({'target_color': color});
  }
}
