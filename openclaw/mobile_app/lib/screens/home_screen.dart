import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/app_provider.dart';
import '../providers/trigger_provider.dart';
import '../providers/config_provider.dart';
import '../utils/theme.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<AppProvider>().checkHealth();
      context.read<ConfigProvider>().loadConfig();
      context.read<ConfigProvider>().loadStats();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('OpenClaw'),
        actions: [
          Consumer<AppProvider>(
            builder: (context, app, _) {
              return IconButton(
                icon: Icon(
                  app.isConnected ? Icons.cloud_done : Icons.cloud_off,
                  color: app.isConnected ? AppTheme.successColor : AppTheme.errorColor,
                ),
                onPressed: () => app.checkHealth(),
              );
            },
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: () async {
          await context.read<AppProvider>().checkHealth();
          await context.read<ConfigProvider>().loadStats();
        },
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildStatusCard(),
              const SizedBox(height: 16),
              _buildQuickActions(),
              const SizedBox(height: 16),
              _buildCurrentConfig(),
              const SizedBox(height: 16),
              _buildStats(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatusCard() {
    return Consumer<AppProvider>(
      builder: (context, app, _) {
        return Card(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                Icon(
                  app.isConnected ? Icons.verified : Icons.warning,
                  size: 48,
                  color: app.isConnected ? AppTheme.successColor : AppTheme.warningColor,
                ),
                const SizedBox(height: 12),
                Text(
                  app.isConnected ? 'Connected' : 'Disconnected',
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    color: app.isConnected ? AppTheme.successColor : AppTheme.warningColor,
                  ),
                ),
                if (app.health.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  Text(
                    'Version: ${app.health['version'] ?? 'Unknown'}',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildQuickActions() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Quick Actions',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: Consumer<TriggerProvider>(
                builder: (context, trigger, _) {
                  return ElevatedButton.icon(
                    onPressed: trigger.isLoading ? null : () => trigger.trigger(),
                    icon: trigger.isLoading
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.flash_on),
                    label: const Text('Trigger'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.primaryColor,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  );
                },
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Consumer<TriggerProvider>(
                builder: (context, trigger, _) {
                  return OutlinedButton.icon(
                    onPressed: () {
                      final result = trigger.lastResult;
                      if (result != null) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text(
                              result['triggered'] == true
                                  ? 'Triggered!'
                                  : 'Not triggered',
                            ),
                            backgroundColor: result['triggered'] == true
                                ? AppTheme.successColor
                                : AppTheme.warningColor,
                          ),
                        );
                      }
                    },
                    icon: const Icon(Icons.info_outline),
                    label: const Text('Last Result'),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildCurrentConfig() {
    return Consumer<ConfigProvider>(
      builder: (context, config, _) {
        return Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Current Configuration',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    IconButton(
                      icon: const Icon(Icons.refresh),
                      onPressed: () => config.loadConfig(),
                    ),
                  ],
                ),
                const Divider(),
                _buildConfigRow('Mode', config.mode.toUpperCase()),
                _buildConfigRow('Polling', config.polling ? 'Enabled' : 'Disabled'),
                _buildConfigRow('Poll Interval', '${config.pollInterval}s'),
                _buildConfigRow('Action', config.action),
                _buildConfigRow('Action Delay', '${config.actionDelay}s'),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildConfigRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Colors.grey)),
          Text(value, style: const TextStyle(fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  Widget _buildStats() {
    return Consumer<ConfigProvider>(
      builder: (context, config, _) {
        final stats = config.stats;
        return Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Statistics',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    IconButton(
                      icon: const Icon(Icons.refresh),
                      onPressed: () => config.loadStats(),
                    ),
                  ],
                ),
                const Divider(),
                Row(
                  children: [
                    Expanded(
                      child: _buildStatItem(
                        'Total',
                        '${stats['total'] ?? 0}',
                        AppTheme.primaryColor,
                      ),
                    ),
                    Expanded(
                      child: _buildStatItem(
                        'Triggered',
                        '${stats['triggered'] ?? 0}',
                        AppTheme.successColor,
                      ),
                    ),
                    Expanded(
                      child: _buildStatItem(
                        'Failed',
                        '${stats['failed'] ?? 0}',
                        AppTheme.errorColor,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildStatItem(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          value,
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
        Text(
          label,
          style: const TextStyle(color: Colors.grey, fontSize: 12),
        ),
      ],
    );
  }
}
