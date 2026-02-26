import 'package:flutter/material.dart';
import '../services/api_service.dart';

class TriggerProvider extends ChangeNotifier {
  final ApiService _api = ApiService();

  bool _isLoading = false;
  bool _triggered = false;
  List<Map<String, dynamic>> _triggers = [];
  Map<String, dynamic>? _lastResult;

  bool get isLoading => _isLoading;
  bool get triggered => _triggered;
  List<Map<String, dynamic>> get triggers => _triggers;
  Map<String, dynamic>? get lastResult => _lastResult;

  Future<void> trigger() async {
    _isLoading = true;
    notifyListeners();

    try {
      _lastResult = await _api.trigger();
      _triggered = _lastResult?['triggered'] ?? false;
    } catch (e) {
      _triggered = false;
    }

    _isLoading = false;
    notifyListeners();
  }

  Future<void> loadTriggers() async {
    _isLoading = true;
    notifyListeners();

    try {
      final result = await _api.getTriggers();
      _triggers = List<Map<String, dynamic>>.from(result);
    } catch (e) {
      _triggers = [];
    }

    _isLoading = false;
    notifyListeners();
  }

  Future<bool> createTrigger(Map<String, dynamic> trigger) async {
    _isLoading = true;
    notifyListeners();

    final result = await _api.createTrigger(trigger);

    _isLoading = false;
    notifyListeners();

    if (result != null) {
      _triggers.add(result);
      notifyListeners();
      return true;
    }
    return false;
  }

  Future<bool> deleteTrigger(String id) async {
    final success = await _api.deleteTrigger(id);
    if (success) {
      _triggers.removeWhere((t) => t['id'] == id);
      notifyListeners();
    }
    return success;
  }

  Future<bool> executeTrigger(String id) async {
    _isLoading = true;
    notifyListeners();

    final result = await _api.executeTrigger(id);
    _lastResult = result;

    _isLoading = false;
    notifyListeners();

    return result?['result'] ?? false;
  }
}
