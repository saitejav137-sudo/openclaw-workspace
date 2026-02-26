import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:fl_chart/fl_chart.dart';
import '../providers/config_provider.dart';
import '../utils/theme.dart';

class StatsScreen extends StatefulWidget {
  const StatsScreen({super.key});

  @override
  State<StatsScreen> createState() => _StatsScreenState();
}

class _StatsScreenState extends State<StatsScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<ConfigProvider>().loadStats();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Statistics'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => context.read<ConfigProvider>().loadStats(),
          ),
        ],
      ),
      body: Consumer<ConfigProvider>(
        builder: (context, config, _) {
          final stats = config.stats;

          if (config.isLoading) {
            return const Center(child: CircularProgressIndicator());
          }

          return RefreshIndicator(
            onRefresh: () => config.loadStats(),
            child: ListView(
              padding: const EdgeInsets.all(16),
              children: [
                _buildOverview(stats),
                const SizedBox(height: 16),
                _buildChart(stats),
                const SizedBox(height: 16),
                _buildDetails(stats),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildOverview(Map<String, dynamic> stats) {
    final total = stats['total'] ?? 0;
    final triggered = stats['triggered'] ?? 0;
    final failed = stats['failed'] ?? 0;
    final successRate = stats['success_rate'] ?? 0.0;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildStatBox('Total', total.toString(), AppTheme.primaryColor),
                _buildStatBox('Triggered', triggered.toString(), AppTheme.successColor),
                _buildStatBox('Failed', failed.toString(), AppTheme.errorColor),
              ],
            ),
            const SizedBox(height: 16),
            Text(
              'Success Rate',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 8),
            Text(
              '${successRate.toStringAsFixed(1)}%',
              style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                color: successRate > 80 ? AppTheme.successColor : AppTheme.warningColor,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatBox(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          value,
          style: TextStyle(
            fontSize: 28,
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

  Widget _buildChart(Map<String, dynamic> stats) {
    final triggered = (stats['triggered'] ?? 0).toDouble();
    final failed = (stats['failed'] ?? 0).toDouble();
    final total = triggered + failed;

    if (total == 0) {
      return Card(
        child: Container(
          height: 200,
          padding: const EdgeInsets.all(16),
          child: const Center(
            child: Text('No data yet'),
          ),
        ),
      );
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Trigger Distribution', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: PieChart(
                PieChartData(
                  sections: [
                    PieChartSectionData(
                      value: triggered,
                      color: AppTheme.successColor,
                      title: '${(triggered / total * 100).toStringAsFixed(0)}%',
                      titleStyle: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                      radius: 80,
                    ),
                    PieChartSectionData(
                      value: failed,
                      color: AppTheme.errorColor,
                      title: '${(failed / total * 100).toStringAsFixed(0)}%',
                      titleStyle: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                      radius: 80,
                    ),
                  ],
                  centerSpaceRadius: 40,
                  sectionsSpace: 2,
                ),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildLegend('Triggered', AppTheme.successColor),
                const SizedBox(width: 24),
                _buildLegend('Failed', AppTheme.errorColor),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLegend(String label, Color color) {
    return Row(
      children: [
        Container(
          width: 12,
          height: 12,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
          ),
        ),
        const SizedBox(width: 8),
        Text(label, style: const TextStyle(color: Colors.grey)),
      ],
    );
  }

  Widget _buildDetails(Map<String, dynamic> stats) {
    final byMode = stats['by_mode'] as Map<String, dynamic>? ?? {};

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('By Mode', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            if (byMode.isEmpty)
              const Center(child: Text('No data'))
            else
              ...byMode.entries.map((entry) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(entry.key.toString().toUpperCase()),
                    Text('${entry.value}'),
                  ],
                ),
              )),
          ],
        ),
      ),
    );
  }
}
