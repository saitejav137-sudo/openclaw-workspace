import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/trigger_provider.dart';
import '../utils/theme.dart';

class TriggersScreen extends StatefulWidget {
  const TriggersScreen({super.key});

  @override
  State<TriggersScreen> createState() => _TriggersScreenState();
}

class _TriggersScreenState extends State<TriggersScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<TriggerProvider>().loadTriggers();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Triggers'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => context.read<TriggerProvider>().loadTriggers(),
          ),
        ],
      ),
      body: Consumer<TriggerProvider>(
        builder: (context, provider, _) {
          if (provider.isLoading) {
            return const Center(child: CircularProgressIndicator());
          }

          if (provider.triggers.isEmpty) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.flash_off, size: 64, color: Colors.grey[600]),
                  const SizedBox(height: 16),
                  const Text('No triggers configured'),
                  const SizedBox(height: 8),
                  ElevatedButton.icon(
                    onPressed: () => _showCreateDialog(context),
                    icon: const Icon(Icons.add),
                    label: const Text('Create Trigger'),
                  ),
                ],
              ),
            );
          }

          return ListView.builder(
            padding: const EdgeInsets.all(16),
            itemCount: provider.triggers.length,
            itemBuilder: (context, index) {
              final trigger = provider.triggers[index];
              return _buildTriggerCard(trigger);
            },
          );
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => _showCreateDialog(context),
        child: const Icon(Icons.add),
      ),
    );
  }

  Widget _buildTriggerCard(Map<String, dynamic> trigger) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: trigger['enabled'] == true
              ? AppTheme.successColor.withOpacity(0.2)
              : Colors.grey.withOpacity(0.2),
          child: Icon(
            Icons.flash_on,
            color: trigger['enabled'] == true ? AppTheme.successColor : Colors.grey,
          ),
        ),
        title: Text(trigger['name'] ?? 'Unnamed'),
        subtitle: Text(trigger['mode'] ?? 'ocr'),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            IconButton(
              icon: const Icon(Icons.play_arrow),
              onPressed: () {
                context.read<TriggerProvider>().executeTrigger(trigger['id']);
              },
            ),
            IconButton(
              icon: const Icon(Icons.delete_outline, color: AppTheme.errorColor),
              onPressed: () {
                _showDeleteDialog(trigger);
              },
            ),
          ],
        ),
      ),
    );
  }

  void _showCreateDialog(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (context) => const _CreateTriggerSheet(),
    );
  }

  void _showDeleteDialog(Map<String, dynamic> trigger) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Trigger'),
        content: Text('Are you sure you want to delete "${trigger['name']}"?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              context.read<TriggerProvider>().deleteTrigger(trigger['id']);
              Navigator.pop(context);
            },
            style: TextButton.styleFrom(foregroundColor: AppTheme.errorColor),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }
}

class _CreateTriggerSheet extends StatefulWidget {
  const _CreateTriggerSheet();

  @override
  State<_CreateTriggerSheet> createState() => _CreateTriggerSheetState();
}

class _CreateTriggerSheetState extends State<_CreateTriggerSheet> {
  final _nameController = TextEditingController();
  final _textController = TextEditingController();
  String _mode = 'ocr';

  @override
  void dispose() {
    _nameController.dispose();
    _textController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom,
        left: 16,
        right: 16,
        top: 16,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Create Trigger', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 16),
          TextField(
            controller: _nameController,
            decoration: const InputDecoration(labelText: 'Name'),
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            value: _mode,
            decoration: const InputDecoration(labelText: 'Mode'),
            items: const [
              DropdownMenuItem(value: 'ocr', child: Text('OCR')),
              DropdownMenuItem(value: 'fuzzy', child: Text('Fuzzy')),
              DropdownMenuItem(value: 'template', child: Text('Template')),
              DropdownMenuItem(value: 'color', child: Text('Color')),
              DropdownMenuItem(value: 'monitor', child: Text('Monitor')),
              DropdownMenuItem(value: 'yolo', child: Text('YOLO')),
            ],
            onChanged: (v) => setState(() => _mode = v!),
          ),
          if (_mode == 'ocr' || _mode == 'fuzzy') ...[
            const SizedBox(height: 12),
            TextField(
              controller: _textController,
              decoration: const InputDecoration(labelText: 'Target Text'),
            ),
          ],
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: _create,
              child: const Text('Create'),
            ),
          ),
          const SizedBox(height: 16),
        ],
      ),
    );
  }

  void _create() {
    final trigger = {
      'name': _nameController.text,
      'mode': _mode,
      'action': 'alt+o',
    };

    if (_mode == 'ocr' || _mode == 'fuzzy') {
      trigger['config'] = {'target_text': _textController.text};
    }

    context.read<TriggerProvider>().createTrigger(trigger);
    Navigator.pop(context);
  }
}
