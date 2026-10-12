// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:slt_mobile_app/main.dart';
import 'package:slt_mobile_app/services/slt_inference_service.dart';
import 'package:slt_mobile_app/shell/history_page.dart';

void main() {
  testWidgets('ATLAS shell shows the splash, then Home with its four entry points', (WidgetTester tester) async {
    await tester.pumpWidget(const SignLanguageTranslationApp(splash: Duration(milliseconds: 300)));
    expect(find.text('Sign Language Translation'), findsOneWidget);
    await tester.pump(const Duration(milliseconds: 400));
    await tester.pumpAndSettle();

    expect(find.text('Hello.'), findsOneWidget);
    for (final entry in ['Start translating', 'Practice', 'Glosses', 'History']) {
      expect(find.text(entry), findsOneWidget);
    }
    expect(find.text('All processing stays on your device.'), findsOneWidget);
  });

  testWidgets('Home opens Practice setup, Glosses and History, and back returns Home', (WidgetTester tester) async {
    await tester.pumpWidget(const SignLanguageTranslationApp(splash: Duration(milliseconds: 1)));
    await tester.pump(const Duration(milliseconds: 10));
    await tester.pumpAndSettle();

    await tester.tap(find.text('Practice'));
    await tester.pumpAndSettle();
    expect(find.text('Choose your\npractice set.'), findsOneWidget);
    expect(find.text('Rotate your phone when practice begins.'), findsOneWidget);
    await tester.tap(find.bySemanticsLabel('Back'));
    await tester.pumpAndSettle();
    expect(find.text('Hello.'), findsOneWidget);

    await tester.tap(find.text('History'));
    await tester.pumpAndSettle();
    expect(find.text('Your conversations.'), findsOneWidget);
    await tester.tap(find.bySemanticsLabel('Back'));
    await tester.pumpAndSettle();

    await tester.tap(find.text('Glosses'));
    await tester.pumpAndSettle();
    expect(find.textContaining('signs to explore.'), findsOneWidget);
  });

  testWidgets('session detail lists recognized signs and the timeline', (WidgetTester tester) async {
    await tester.pumpWidget(MaterialApp(
      home: SessionDetailPage(
        row: const {
          'started_utc': '2026-09-29T01:28:00Z',
          'duration_seconds': 36,
          'glosses': ['HELLO', 'fs-GELO'],
          'sentences': <String>[],
          'complete': false,
          'timeline': [
            {'seconds': 0.4, 'gloss': 'HELLO'},
            {'seconds': 2.1, 'gloss': 'fs-GELO'},
          ],
        },
        onBack: () {},
      ),
    ));
    expect(find.text('HELLO  GELO'), findsOneWidget);
    expect(find.text('Unfinished'), findsOneWidget);
    expect(find.text('This session ended before Finish.'), findsOneWidget);
    expect(find.text('0:02'), findsOneWidget);
  });

  test('history rows are described like the desktop shell', () {
    final (started, length, said) = describeSession({
      'started_utc': '2026-09-28T13:05:00Z',
      'duration_seconds': 75.4,
      'glosses': ['HELLO', 'fs-GELO'],
      'sentences': ['Hello, Gelo.'],
    });
    expect(started, contains('2026'));
    expect(length, '1:15');
    expect(said, 'Hello, Gelo.');
    expect(describeSession({'glosses': ['HELLO', 'HOW']}).$3, 'HELLO HOW');
    expect(describeSession({}).$3, 'No recognized signs');
  });

  test('native inference result contract parses single and phrase output', () {
    final result = SltInferenceResult.fromMap({
      'glosses': ['HELLO', 'GOOD', 'MORNING'],
      'english': 'Hello, good morning.',
      'literalEnglish': 'Hello good morning.',
      'renderingMode': 'reviewed_template',
      'safeFallbackUsed': false,
      'benchmark': {'iterations': 20},
      'reportPath': '/tmp/report.json',
      'diagnosticPath': '/tmp/diagnostics/capture',
    });

    expect(result.glosses, ['HELLO', 'GOOD', 'MORNING']);
    expect(result.english, 'Hello, good morning.');
    expect(result.benchmark?['iterations'], 20);
    expect(result.diagnosticPath, '/tmp/diagnostics/capture');
  });
}
