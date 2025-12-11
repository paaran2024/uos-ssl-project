// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:contact/main.dart';
import 'package:contact/picture_tab.dart';
import 'package:contact/video_tab.dart';

void main() {
  testWidgets('App navigation smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(MyApp());

    // Verify that the initial tab is 'picture'.
    expect(find.text('picture'), findsOneWidget);
    expect(find.text('video'), findsOneWidget);

    // Verify PictureTab is shown
    expect(find.byType(PictureTab), findsOneWidget);
    expect(find.byType(VideoTab), findsNothing);

    // Tap the 'video' tab and trigger a frame.
    await tester.tap(find.byIcon(Icons.video_collection_rounded));
    await tester.pump();

    // Verify that VideoTab is shown
    expect(find.byType(PictureTab), findsNothing);
    expect(find.byType(VideoTab), findsOneWidget);
  });
}
