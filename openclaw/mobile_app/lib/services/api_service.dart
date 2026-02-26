import 'dart:convert';
import 'package:dio/dio.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class ApiService {
  static const String baseUrl = 'http://localhost:8765';
  String? _apiKey;

  late final Dio _dio;
  WebSocketChannel? _wsChannel;

  ApiService() {
    _dio = Dio(BaseOptions(
      baseUrl: baseUrl,
      connectTimeout: const Duration(seconds: 10),
      receiveTimeout: const Duration(seconds: 10),
      headers: {
        'Content-Type': 'application/json',
      },
    ));

    _dio.interceptors.add(InterceptorsWrapper(
      onRequest: (options, handler) {
        if (_apiKey != null) {
          options.queryParameters['api_key'] = _apiKey;
        }
        return handler.next(options);
      },
      onError: (error, handler) {
        print('API Error: ${error.message}');
        return handler.next(error);
      },
    ));
  }

  void setApiKey(String? apiKey) {
    _apiKey = apiKey;
  }

  // Health
  Future<Map<String, dynamic>> getHealth() async {
    try {
      final response = await _dio.get('/health');
      return response.data;
    } catch (e) {
      return {'status': 'error', 'message': e.toString()};
    }
  }

  // Trigger
  Future<Map<String, dynamic>> trigger() async {
    try {
      final response = await _dio.post('/api/trigger');
      return response.data;
    } catch (e) {
      return {'status': 'error', 'message': e.toString()};
    }
  }

  // Config
  Future<Map<String, dynamic>> getConfig() async {
    try {
      final response = await _dio.get('/api/config');
      return response.data;
    } catch (e) {
      return {};
    }
  }

  Future<bool> updateConfig(Map<String, dynamic> config) async {
    try {
      await _dio.put('/api/config', data: config);
      return true;
    } catch (e) {
      return false;
    }
  }

  // Stats
  Future<Map<String, dynamic>> getStats() async {
    try {
      final response = await _dio.get('/api/stats');
      return response.data;
    } catch (e) {
      return {
        'total': 0,
        'triggered': 0,
        'failed': 0,
        'success_rate': 0.0,
      };
    }
  }

  // Triggers CRUD
  Future<List<dynamic>> getTriggers() async {
    try {
      final response = await _dio.get('/api/v1/triggers');
      return response.data;
    } catch (e) {
      return [];
    }
  }

  Future<Map<String, dynamic>?> createTrigger(Map<String, dynamic> trigger) async {
    try {
      final response = await _dio.post('/api/v1/triggers', data: trigger);
      return response.data;
    } catch (e) {
      return null;
    }
  }

  Future<bool> deleteTrigger(String id) async {
    try {
      await _dio.delete('/api/v1/triggers/$id');
      return true;
    } catch (e) {
      return false;
    }
  }

  Future<Map<String, dynamic>?> executeTrigger(String id) async {
    try {
      final response = await _dio.post('/api/v1/triggers/$id/execute');
      return response.data;
    } catch (e) {
      return null;
    }
  }

  // WebSocket
  void connectWebSocket(void Function(dynamic) onMessage) {
    final url = _apiKey != null
        ? 'ws://localhost:8766?api_key=$_apiKey'
        : 'ws://localhost:8766';

    try {
      _wsChannel = WebSocketChannel.connect(Uri.parse(url));
      _wsChannel!.stream.listen(
        onMessage,
        onError: (error) => print('WebSocket Error: $error'),
        onDone: () => print('WebSocket closed'),
      );
    } catch (e) {
      print('WebSocket connection failed: $e');
    }
  }

  void sendWebSocketMessage(Map<String, dynamic> message) {
    _wsChannel?.sink.add(jsonEncode(message));
  }

  void disconnectWebSocket() {
    _wsChannel?.sink.close();
    _wsChannel = null;
  }

  // Screenshot
  Future<String?> getScreenshot() async {
    try {
      final response = await _dio.get(
        '/api/v1/screenshots',
        options: Options(responseType: ResponseType.bytes),
      );
      return base64Encode(response.data);
    } catch (e) {
      return null;
    }
  }
}
