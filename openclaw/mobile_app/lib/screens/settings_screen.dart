import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';
import '../providers/app_provider.dart';
import '../utils/theme.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _serverController = TextEditingController();
  final _apiKeyController = TextEditingController();

  @override
  void initState() {
    super.initState();
    final app = context.read<AppProvider>();
    _serverController.text = app.serverUrl;
    _apiKeyController.text = app.apiKey ?? '';
  }

  @override
  void dispose() {
    _serverController.dispose();
    _apiKeyController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildConnectionSettings(),
          const SizedBox(height: 16),
          _buildAboutSection(),
          const SizedBox(height: 16),
          _buildActionsSection(),
        ],
      ),
    );
  }

  Widget _buildConnectionSettings() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Connection', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 16),
            TextField(
              controller: _serverController,
              decoration: const InputDecoration(
                labelText: 'Server URL',
                prefixIcon: Icon(Icons.link),
              ),
              onChanged: (value) {
                context.read<AppProvider>().setServerUrl(value);
              },
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _apiKeyController,
              decoration: const InputDecoration(
                labelText: 'API Key',
                prefixIcon: Icon(Icons.key),
              ),
              obscureText: true,
              onChanged: (value) {
                context.read<AppProvider>().setApiKey(value.isEmpty ? null : value);
              },
            ),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: Consumer<AppProvider>(
                builder: (context, app, _) {
                  return ElevatedButton.icon(
                    onPressed: () => app.checkHealth(),
                    icon: const Icon(Icons.refresh),
                    label: const Text('Test Connection'),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAboutSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('About', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 16),
            ListTile(
              leading: const Icon(Icons.info_outline),
              title: const Text('OpenClaw'),
              subtitle: const Text('Version 2.0.0'),
              contentPadding: EdgeInsets.zero,
            ),
            ListTile(
              leading: const Icon(Icons.code),
              title: const Text('GitHub'),
              subtitle: const Text('View source code'),
              trailing: const Icon(Icons.open_in_new, size: 18),
              contentPadding: EdgeInsets.zero,
              onTap: () => _launchUrl('https://github.com/saitejav137-sudo/openclaw-workspace'),
            ),
            ListTile(
              leading: const Icon(Icons.description),
              title: const Text('Documentation'),
              subtitle: const Text('API docs & guides'),
              trailing: const Icon(Icons.open_in_new, size: 18),
              contentPadding: EdgeInsets.zero,
              onTap: () => _launchUrl('http://localhost:8765/api-docs'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionsSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Actions', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 16),
            ListTile(
              leading: const Icon(Icons.dashboard, color: AppTheme.primaryColor),
              title: const Text('Web Dashboard'),
              subtitle: const Text('Open in browser'),
              trailing: const Icon(Icons.open_in_new, size: 18),
              contentPadding: EdgeInsets.zero,
              onTap: () => _launchUrl('http://localhost:8765/dashboard'),
            ),
            ListTile(
              leading: const Icon(Icons.edit_note, color: AppTheme.primaryColor),
              title: const Text('Config Editor'),
              subtitle: const Text('Visual configuration'),
              trailing: const Icon(Icons.open_in_new, size: 18),
              contentPadding: EdgeInsets.zero,
              onTap: () => _launchUrl('http://localhost:8765/config-editor'),
            ),
            ListTile(
              leading: const Icon(Icons.terminal, color: AppTheme.primaryColor),
              title: const Text('API Documentation'),
              subtitle: const Text('OpenAPI/Swagger'),
              trailing: const Icon(Icons.open_in_new, size: 18),
              contentPadding: EdgeInsets.zero,
              onTap: () => _launchUrl('http://localhost:8765/api-docs'),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _launchUrl(String url) async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri);
    }
  }
}
