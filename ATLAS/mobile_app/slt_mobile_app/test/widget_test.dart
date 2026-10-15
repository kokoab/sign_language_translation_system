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
import 'package:slt_mobile_app/shell/manual_page.dart';
import 'package:slt_mobile_app/shell/practice_page.dart';

/// A 390 × 844 portrait phone.
void _phone(WidgetTester tester) {
  tester.view.physicalSize = const Size(1170, 2532);
  tester.view.devicePixelRatio = 3;
  addTearDown(tester.view.reset);
}

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
    expect(find.text('Landscape recommended. Portrait works too.'), findsOneWidget);
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

  testWidgets('Home opens the manual; a section leads to the next and back returns Home', (WidgetTester tester) async {
    _phone(tester);
    await tester.pumpWidget(const SignLanguageTranslationApp(splash: Duration(milliseconds: 1)));
    await tester.pump(const Duration(milliseconds: 10));
    await tester.pumpAndSettle();

    await tester.tap(find.text('How to use ATLAS'));
    await tester.pumpAndSettle();
    expect(find.text('Your guide\nto ATLAS.'), findsOneWidget);

    await tester.tap(find.text('Gestures and special signs'));
    await tester.pumpAndSettle();
    expect(find.text('Finish pose'), findsOneWidget);
    await tester.scrollUntilVisible(find.text('Next: Practice'), 300);
    await tester.tap(find.text('Next: Practice'));
    await tester.pumpAndSettle();
    expect(find.text('Learn signs with instant feedback.'), findsOneWidget);

    await tester.tap(find.bySemanticsLabel('Back'));
    await tester.pumpAndSettle();
    expect(find.text('Your guide\nto ATLAS.'), findsOneWidget);
    await tester.tap(find.bySemanticsLabel('Back'));
    await tester.pumpAndSettle();
    expect(find.text('Hello.'), findsOneWidget);
  });

  testWidgets('every manual section lays out on a phone', (WidgetTester tester) async {
    _phone(tester);
    for (final section in manualSections) {
      await tester.pumpWidget(MaterialApp(
        home: ManualSectionPage(
          key: ValueKey(section.id),
          section: section,
          onOpen: (_) {},
          onWatch: (_) {},
          onBack: () {},
        ),
      ));
      // Build every block, not only the first screenful.
      final list = find.byType(Scrollable).first;
      for (var i = 0; i < 40; i++) {
        await tester.drag(list, const Offset(0, -400));
        await tester.pump();
      }
      expect(tester.takeException(), isNull, reason: section.id);
      expect(find.textContaining(manualSections.last == section ? 'Back to the guide' : 'Next: '), findsOneWidget,
          reason: section.id);
    }
    expect(manualSection('screen')?.title, 'Reading the screen');
    expect(manualSection('practice')?.title, 'Practice');
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

  testWidgets('session detail shows the English first, speaks it again and deletes after confirming',
      (WidgetTester tester) async {
    final spoken = <String>[];
    var deleted = 0;
    await tester.pumpWidget(MaterialApp(
      home: SessionDetailPage(
        row: const {
          'started_utc': '2026-10-15T01:28:00Z',
          'duration_seconds': 12,
          'glosses': ['HELLO', 'HOW', 'YOU'],
          'sentences': ['Hello, how are you?'],
          'complete': true,
          'timeline': <Map<String, Object>>[],
        },
        onBack: () {},
        onSpeak: spoken.add,
        onDelete: () => deleted++,
      ),
    ));
    final english = tester.getTopLeft(find.text('Hello, how are you?'));
    final signs = tester.getTopLeft(find.text('HELLO  HOW  YOU'));
    expect(english.dy, lessThan(signs.dy));
    expect(find.text('Copy'), findsNothing);
    await tester.tap(find.text('Speak again'));
    expect(spoken, ['Hello, how are you?']);
    await tester.tap(find.text('Delete session'));
    await tester.pumpAndSettle();
    expect(find.text('Delete this session?'), findsOneWidget);
    await tester.tap(find.text('Cancel'));
    await tester.pumpAndSettle();
    expect(deleted, 0);
    await tester.tap(find.text('Delete session'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Delete'));
    await tester.pumpAndSettle();
    expect(deleted, 1);
  });

  testWidgets('history clears all sessions after confirming', (WidgetTester tester) async {
    var cleared = 0;
    await tester.pumpWidget(MaterialApp(
      home: HistoryPage(
        sessions: const [{'started_utc': '2026-10-15T01:28:00Z', 'sentences': ['Hello.'], 'glosses': ['HELLO']}],
        onSelect: (_) {},
        onBack: () {},
        onClearAll: () => cleared++,
      ),
    ));
    await tester.tap(find.text('Clear all'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Delete'));
    await tester.pumpAndSettle();
    expect(cleared, 1);
  });

  testWidgets('practice result opens a missed sign', (WidgetTester tester) async {
    final watched = <String>[];
    await tester.pumpWidget(MaterialApp(
      home: PracticeResultPage(
        result: const PracticeResult(score: 3, total: 5, missed: ['HELLO', 'THANK-YOU']),
        onAgain: () {},
        onDone: () {},
        onWatch: watched.add,
      ),
    ));
    expect(find.text('Tap a sign to watch it again.'), findsOneWidget);
    await tester.tap(find.text('THANK-YOU'));
    expect(watched, ['THANK-YOU']);
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
