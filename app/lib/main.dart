import 'package:flutter/material.dart';
import 'picture_tab.dart';
import 'video_tab.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  int _currentIndex = 0;

  @override
  Widget build(BuildContext context) {
    final List<Widget> screens = [
      PictureTab(),
      VideoTab(),
    ];

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        body: screens[_currentIndex],

        bottomNavigationBar: BottomNavigationBar(
          currentIndex: _currentIndex,
          onTap: (index) => setState(() => _currentIndex = index),
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.image),
              label: "picture",
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.video_collection_rounded),
              label: "video",
            ),
          ],
        ),
      ),
    );
  }
}
