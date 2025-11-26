import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:emutest/media_store_saver.dart';

class PictureTab extends StatefulWidget {
  const PictureTab({super.key});

  @override
  State<PictureTab> createState() => _PictureTabState();
}

class _PictureTabState extends State<PictureTab> {
  Uint8List? _inputBytes;
  Uint8List? _outputBytes;
  String _inferenceTime = "";

  Future<void> _pickImage() async {
    try {
      final ImagePicker picker = ImagePicker();
      final XFile? picked = await picker.pickImage(
        source: ImageSource.gallery,
      );

      if (picked != null) {
        final bytes = await picked.readAsBytes();
        setState(() {
          _inputBytes = bytes;
          _outputBytes = null;
          _inferenceTime = "";
        });
      }
    } catch (e) {
      print("이미지 선택 에러: $e");
    }
  }

  Future<void> _convertImage() async {
    if (_inputBytes == null) return;

    final stopwatch = Stopwatch()..start();

    // 이미지 변환 로직
    await Future.delayed(const Duration(milliseconds: 300));
    _outputBytes = _inputBytes;

    stopwatch.stop();

    setState(() {
      _inferenceTime = "${stopwatch.elapsedMilliseconds} ms";
    });
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: Column(
        children: [
          const SizedBox(height: 15),
          Row(
            children: [
              // 입력 이미지
              Expanded(
                child: GestureDetector(
                  onTap: _pickImage,
                  child: Container(
                    height: 300,
                    margin: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade300,
                      shape: BoxShape.rectangle,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: _inputBytes == null
                        ? Center(
                            child: Container(
                              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                              decoration: BoxDecoration(
                                color: Colors.grey,
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: const Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(Icons.file_upload_outlined, color: Colors.white, size: 24),
                                  SizedBox(width: 8),
                                  Text(
                                    "image upload",
                                    style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 16),
                                  ),
                                ],
                              ),
                            ),
                          )
                        : ClipRRect(
                            borderRadius: BorderRadius.circular(10),
                            child: Image.memory(_inputBytes!, fit: BoxFit.cover),
                          ),
                  ),
                ),
              ),

              // 출력 이미지
              Expanded(
                child: Container(
                  height: 300,
                  margin: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade300,
                    shape: BoxShape.rectangle,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: _outputBytes == null
                      ? const Center(child: Text("output image"))
                      : ClipRRect(
                          borderRadius: BorderRadius.circular(10),
                          child: Image.memory(_outputBytes!, fit: BoxFit.cover),
                        ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 20),

          // 버튼 전환 로직
          if (_outputBytes == null)
            ElevatedButton(
              onPressed: _convertImage,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.orange,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.cached_rounded),
                  SizedBox(width: 8),
                  Text("image upscaling", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                ],
              ),
            )
          else
            ElevatedButton(
              onPressed: () async {
                bool ok = await MediaStoreSaver.saveImage(_outputBytes!);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text(ok ? "갤러리에 저장됨!" : "저장 실패")),
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor:Colors.orange,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.download_rounded),
                  SizedBox(width: 8),
                  Text("image download", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                ],
              ),
            ),

          const SizedBox(height: 16),

          Text(
            "inference time: $_inferenceTime",
            style: const TextStyle(fontSize: 16),
          ),

          const SizedBox(height: 20),
        ],
      ),
    );
  }
}
