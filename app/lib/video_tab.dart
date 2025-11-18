import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';

class VideoTab extends StatefulWidget {
  @override
  _VideoTabState createState() => _VideoTabState();
}

class _VideoTabState extends State<VideoTab> {
  Uint8List? _inputVideoBytes;
  Uint8List? _outputVideoBytes;

  // 비디오 선택
  Future<void> _pickVideo() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.video,
      withData: true,
    );

    if (result != null) {
      setState(() {
        _inputVideoBytes = result.files.first.bytes;
        _outputVideoBytes = null;
      });

      _processVideo();
    }
  }

  // AI 처리 (더미)
  Future<void> _processVideo() async {
    await Future.delayed(Duration(seconds: 2));
    setState(() {
      _outputVideoBytes = _inputVideoBytes;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [

          // 입력 비디오
          Text("입력 비디오", style: TextStyle(fontSize: 18)),
          SizedBox(height: 10),

          GestureDetector(
            onTap: _pickVideo,
            child: Container(
              height: 250,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey.shade300,   // 동일 스타일
                borderRadius: BorderRadius.circular(10),
              ),
              child: _inputVideoBytes == null
                  ? Center(child: Text("비디오 선택"))
                  : Center(child: Text("입력 비디오 로드됨")),
            ),
          ),

          SizedBox(height: 40),

          // 출력 비디오
          Text("출력 비디오", style: TextStyle(fontSize: 18)),
          SizedBox(height: 10),

          Container(
            height: 250,
            width: double.infinity,
            decoration: BoxDecoration(
              color: Colors.grey.shade300,   // 동일 스타일
              borderRadius: BorderRadius.circular(10),
            ),
            child: _outputVideoBytes == null
                ? Center(child: Text("출력 비디오"))
                : Center(child: Text("복원된 비디오 로드됨")),
          ),
        ],
      ),
    );
  }
}