import 'package:flutter/material.dart';
import 'picture_tab.dart';
import 'video_tab.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text("Ondevice"),
          centerTitle: false,
          bottom: PreferredSize(
            preferredSize: Size.fromHeight(1),
            child: Container(
              color: Colors.grey.shade300,   // 구분선 색
              height: 1,
            ),
          ),
        ),


        bottomNavigationBar: const TabBar(
          labelColor: Colors.blue,
          unselectedLabelColor: Colors.grey,
          tabs: [
            Tab(text: "사진", icon: Icon(Icons.photo)),
            Tab(text: "비디오", icon: Icon(Icons.video_library)),
          ],
        ),

        body: const TabBarView(
          children: [
            PictureTab(),
            VideoTab(),
          ],
        ),
      ),
    );
  }
}
