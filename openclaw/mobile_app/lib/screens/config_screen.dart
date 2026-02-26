import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/config_provider.dart';
import '../utils/theme.dart';

class ConfigScreen extends StatefulWidget {
  const ConfigScreen({super.key});

  @override
  State<ConfigScreen> createState() => _ConfigScreenState();
}

class _ConfigScreenState extends State<ConfigScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<ConfigProvider>().loadConfig();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Configuration'),
        actions: [
          IconButton(
            icon: const Icon(Icons.save),
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Config saved!')),
              );
            },
          ),
        ],
      ),
      body: Consumer<ConfigProvider>(
        builder: (context, config, _) {
          if (config.isLoading) {
            return const Center(child: CircularProgressIndicator());
          }

          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              _buildModeSection(config),
              const SizedBox(height: 16),
              _buildPollingSection(config),
              const SizedBox(height: 16),
              _buildActionSection(config),
              const SizedBox(height: 16),
              _buildDetectionSection(config),
            ],
          );
        },
      ),
    );
  }

  Widget _buildModeSection(ConfigProvider config) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Detection Mode', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: ['ocr', 'fuzzy', 'template', 'color', 'monitor', 'yolo', 'window'].map((mode) {
                final isSelected = config.mode == mode;
                return ChoiceChip(
                  label: Text(mode.toUpperCase()),
                  selected: isSelected,
                  onSelected: (_) => config.setMode(mode),
                  selectedColor: AppTheme.primaryColor,
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPollingSection(ConfigProvider config) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Polling', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            SwitchListTile(
              title: const Text('Enable Polling'),
              value: config.polling,
              onChanged: config.setPolling,
              contentPadding: EdgeInsets.zero,
            ),
            if (config.polling)
              Column(
                children: [
                  Text('Interval: ${config.pollInterval}s'),
                  Slider(
                    value: config.pollInterval,
                    min: 0.1,
                    max: 5.0,
                    divisions: 49,
                    onChanged: (v) => config.setPollInterval(v),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionSection(ConfigProvider config) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Action', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            TextField(
              decoration: const InputDecoration(labelText: 'Keyboard Action'),
              controller: TextEditingController(text: config.action),
              onChanged: (v) => config.setAction(v),
            ),
            const SizedBox(height: 12),
            Text('Delay: ${config.actionDelay}s'),
            Slider(
              value: config.actionDelay,
              min: 0.1,
              max: 10.0,
              divisions: 99,
              onChanged: (v) => config.setActionDelay(v),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDetectionSection(ConfigProvider config) {
    if (config.mode != 'ocr' && config.mode != 'fuzzy') {
      return const SizedBox();
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Detection Settings', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            TextField(
              decoration: const InputDecoration(labelText: 'Target Text'),
              onChanged: (v) => config.setTargetText(v),
            ),
          ],
        ),
      ),
    );
  }
}
