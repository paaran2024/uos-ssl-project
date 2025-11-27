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
      debugShowCheckedModeBanner: false, // 디버그 띠 제거
      theme: ThemeData(
        primarySwatch: Colors.orange,
        scaffoldBackgroundColor: Colors.white,
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.white,
          elevation: 0,
          iconTheme: IconThemeData(color: Colors.black),
          titleTextStyle: TextStyle(color: Colors.black, fontSize: 20),
        ),
      ),
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
          title: RichText(
            text: const TextSpan(
              style: TextStyle(
                fontSize: 25,
                fontWeight: FontWeight.w900,
                fontFamily: 'Roboto',
              ),
              children: [
                TextSpan(
                  text: "Cata",
                  style: TextStyle(color: Colors.orange),
                ),
                TextSpan(
                  text: "Scaler",
                  style: TextStyle(color: Colors.black),
                ),
              ],
            ),
          ),
          centerTitle: false,
          bottom: PreferredSize(
            preferredSize: const Size.fromHeight(1),
            child: Container(color: Colors.grey.shade300, height: 1),
          ),
        ),

        bottomNavigationBar: Container(
          decoration: const BoxDecoration(
            border: Border(top: BorderSide(color: Colors.grey, width: 0.5)),
          ),
          child: const TabBar(
            indicatorColor: Colors.cyan,
            labelColor: Colors.cyan,
            unselectedLabelColor: Colors.grey,
            tabs: [
              Tab(text: "picture", icon: Icon(Icons.image_outlined)),
              Tab(text: "video", icon: Icon(Icons.play_circle_outline)),
            ],
          ),
        ),

        body: const TabBarView(
          physics: NeverScrollableScrollPhysics(), // 스와이프 방지
          children: [PictureTab(), VideoTab()],
        ),
      ),
    );
  }
}
