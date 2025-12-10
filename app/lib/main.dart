import 'package:flutter/material.dart';

import 'constants.dart';
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
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.orange,
        scaffoldBackgroundColor: AppConstants.backgroundColor,
        appBarTheme: const AppBarTheme(
          backgroundColor: AppConstants.backgroundColor,
          elevation: 0,
          iconTheme: IconThemeData(color: Colors.black),
          titleTextStyle: TextStyle(color: Colors.black, fontSize: 20),
        ),
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: _buildAppTitle(),
          centerTitle: false,
          bottom: PreferredSize(
            preferredSize: const Size.fromHeight(1),
            child: Container(color: Colors.grey.shade300, height: 1),
          ),
        ),
        bottomNavigationBar: _buildBottomTabBar(),
        body: const TabBarView(
          physics: NeverScrollableScrollPhysics(),
          children: [PictureTab(), VideoTab()],
        ),
      ),
    );
  }

  Widget _buildAppTitle() {
    return RichText(
      text: const TextSpan(
        style: TextStyle(
          fontSize: 25,
          fontWeight: FontWeight.w900,
          fontFamily: 'Roboto',
        ),
        children: [
          TextSpan(
            text: "Cata",
            style: TextStyle(color: AppConstants.primaryColor),
          ),
          TextSpan(
            text: "Scaler",
            style: TextStyle(color: Colors.black),
          ),
        ],
      ),
    );
  }

  Widget _buildBottomTabBar() {
    return Container(
      decoration: const BoxDecoration(
        border: Border(top: BorderSide(color: Colors.grey, width: 0.5)),
      ),
      child: const TabBar(
        indicatorColor: AppConstants.accentColor,
        labelColor: AppConstants.accentColor,
        unselectedLabelColor: Colors.grey,
        tabs: [
          Tab(text: "picture", icon: Icon(Icons.image_outlined)),
          Tab(text: "video", icon: Icon(Icons.play_circle_outline)),
        ],
      ),
    );
  }
}
