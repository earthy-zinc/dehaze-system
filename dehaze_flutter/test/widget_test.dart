import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:dehaze_flutter/app/app.dart';

void main() {
  testWidgets('App smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const App());

    // Verify that the app builds without errors
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
