import 'package:flutter/material.dart';
import '../services/api_service.dart';

class AppProvider extends ChangeNotifier {
  final ApiService _api = ApiService();

  bool _isLoading = false;
  bool _isConnected = false;
  String _serverUrl = 'http://localhost:8765';
  String? _apiKey;
  Map<String, dynamic> _health = {};

  bool get isLoading => _isLoading;
  bool get isConnected => _isConnected;
  String get serverUrl => _serverUrl;
  String? get apiKey => _apiKey;
  Map<String, dynamic> get health => _health;
  ApiService get api => _api;

  void setServerUrl(String url) {
    _serverUrl = url;
    notifyListeners();
  }

  void setApiKey(String? key) {
    _apiKey = key;
    _api.setApiKey(key);
    notifyListeners();
  }

  Future<void> checkHealth() async {
    _isLoading = true;
    notifyListeners();

    try {
      _health = await _api.getHealth();
      _isConnected = _health['status'] == 'healthy';
    } catch (e) {
      _isConnected = false;
    }

    _isLoading = false;
    notifyListeners();
  }

  Future<void> connect() async {
    await checkHealth();
    if (_isConnected) {
      _api.connectWebSocket((message) {
        print('WebSocket: $message');
      });
    }
  }

  void disconnect() {
    _api.disconnectWebSocket();
    _isConnected = false;
    notifyListeners();
  }
}
