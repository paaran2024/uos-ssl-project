import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';

class PictureTab extends StatefulWidget {
  @override
  _PictureTabState createState() => _PictureTabState();
}

class _PictureTabState extends State<PictureTab> {
  Uint8List? _inputBytes;
  Uint8List? _outputBytes;
  String _inferenceTime = "";

  // 웹/PC/모바일 공통 이미지 선택
  Future<void> _pickImageWeb() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      withData: true,
    );

    if (result != null) {
      setState(() {
        _inputBytes = result.files.first.bytes;
        _outputBytes = null;
        _inferenceTime = "";
      });

      _runSuperResolution();
    }
  }

  // 더미 복원 처리
  Future<void> _runSuperResolution() async {
    if (_inputBytes == null) return;

    final stopwatch = Stopwatch()..start();

    // AI 로직 들어가는 곳
    await Future.delayed(Duration(seconds: 2));

    stopwatch.stop();

    setState(() {
      _outputBytes = _inputBytes; // 임시
      _inferenceTime =
      "${(stopwatch.elapsedMilliseconds / 1000).toStringAsFixed(3)} sec";
    });
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [

          Row(
            children: [
              Expanded(
                child: Column(
                  children: [
                    Text("입력 이미지", style: TextStyle(fontSize: 18)),
                    SizedBox(height: 10),

                    GestureDetector(
                      onTap: _pickImageWeb,
                      child: Container(
                        height: 300,
                        decoration: BoxDecoration(
                          color: Colors.grey.shade300,   // 회색 배경
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: _inputBytes == null
                            ? Center(child: Text("이미지 선택"))
                            : Image.memory(_inputBytes!, fit: BoxFit.cover),
                      ),
                    ),
                  ],
                ),
              ),

              SizedBox(width: 20),

              Expanded(
                child: Column(
                  children: [
                    Text("출력 이미지", style: TextStyle(fontSize: 18)),
                    SizedBox(height: 10),

                    Container(
                      height: 300,
                      decoration: BoxDecoration(
                        color: Colors.grey.shade300,   // 회색 배경
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: _outputBytes == null
                          ? Center(child: Text("출력 이미지"))
                          : Image.memory(_outputBytes!, fit: BoxFit.cover),
                    ),

                  ],
                ),
              ),
            ],
          ),

          Expanded(
            child: Center(
              child: Text(
                _inferenceTime.isEmpty
                    ? "Inference time: -"
                    : "Inference time: $_inferenceTime",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
